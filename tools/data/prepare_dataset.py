# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL @2022
import argparse
import os
import os.path as osp
from multiprocessing import Pool

import cv2
import lmdb
import mmcv
import numpy as np
from tqdm import tqdm


def crop_one_image(path, opt, ext='.png'):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is smaller
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, _ = osp.splitext(osp.basename(path))

    img = mmcv.imread(path, flag='unchanged')

    if img.ndim == 2 or img.ndim == 3:
        h, w = img.shape[:2]
    else:
        raise ValueError(f'Image ndim should be 2 or 3, but got {img.ndim}.')

    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            patch = img[x:x + crop_size, y:y + crop_size, ...]
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_{index:d}{ext}'),
                patch, [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


def crop_patches(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    img_list = list(mmcv.scandir(input_folder))
    img_list = [osp.join(input_folder, v) for v in img_list]

    prog_bar = tqdm(total=len(img_list), ncols=0)
    pool = Pool(opt['n_thread'])
    for path in img_list:
        # for debugging
        # process_info = crop_one_image(path, opt)
        # prog_bar.update()
        pool.apply_async(crop_one_image,
                         args=(path, opt),
                         callback=lambda _: prog_bar.update())
    pool.close()
    pool.join()
    prog_bar.close()
    print('Patching done.')


def main_crop_patches(args, input_folder, save_folder):
    """A multi-thread tool to crop large images to sub-images for faster IO.

    opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9.
            A higher value means a smaller size and longer compression time.
            Use 0 for faster CPU decompression. Default: 3, same in cv2.
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower
            than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        After process, each sub_folder should have the same number of
        subimages. You can also specify scales by modifying the argument
        'scales'. Remember to modify opt configurations according to your
        settings.
    """

    opt = {}
    opt['n_thread'] = args.nthreads
    opt['compression_level'] = args.compression_level
    opt['crop_size'] = args.patch_size
    opt['step'] = args.step
    opt['thresh_size'] = args.thresh_size
    opt['input_folder'] = input_folder
    opt['save_folder'] = save_folder
    crop_patches(opt)


def read_img_worker(img_path, compress_level, key):
    """Read image worker.

    Args:
        img_path (str): Image path.
        compress_level (int): Compress level when encoding images.

    Returns:
        str: Image key.
        byte: Image byte.
        tuple[int]: Image shape.
    """
    img = mmcv.imread(img_path, flag='unchanged')
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return img_path, img_byte, key


def make_lmdb(data_path,
              lmdb_path,
              img_names,
              batch=5000,
              compress_level=1,
              multiprocessing_read=False,
              n_thread=40,
              meta_name='meta_info.txt'):
    """Make lmdb.

    Contents of lmdb. The file structure is:
    example.lmdb
    `-- data.mdb
    `-- lock.mdb
    `-- meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records the image name (with extension).

    LMDB key: LMDB root / image name
    It is because the input of the LoadImageFromFile pipeline is also used for
    LMDB key search, which is also the results[f'{self.key}_path'].
    According to the PairedSameSizeImageDataset dataset,
    {gt,lq}_path is defined as osp.join(self.{gt,lq}_folder, name), where the
    name is listed in meta_info.txt.

    If `multiprocessing_read` is True, it will read all the images to memory
    using multiprocessing. Thus, your server needs to have enough memory.

    Args:
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        img_names (str): Image name list.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
        multiprocessing_read (bool): Whether use multiprocessing to read all
            the images to memory. Default: False.
        n_thread (int): For multiprocessing.
    """
    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    print(f'Total images: {len(img_names)}')
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("'lmdb_path' must end with '.lmdb'.")

    if multiprocessing_read:
        # read all the images to memory (multiprocessing)
        dataset = {}  # use dict to keep the order for multiprocessing
        print(f'Read images with multiprocessing, #thread: {n_thread} ...')
        prog_bar = mmcv.ProgressBar(len(img_names))

        def callback(img_path, img_byte, key):
            """get the image data and update prog_bar."""
            dataset[key] = img_byte
            prog_bar.update()

        pool = Pool(n_thread)
        for img_name in img_names:
            img_path = osp.join(data_path, img_name)
            key = osp.join(lmdb_path, img_name)
            pool.apply_async(read_img_worker,
                             args=(img_path, compress_level, key),
                             callback=callback)
        pool.close()
        pool.join()

    # create lmdb environment
    # obtain data size for one image
    img = mmcv.imread(osp.join(data_path, img_names[0]), flag='unchanged')
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    data_size_per_img = img_byte.nbytes
    print('Data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(img_names)
    env = lmdb.open(lmdb_path, map_size=data_size * 10)

    # write data to lmdb
    prog_bar = tqdm(total=len(img_names), ncols=0)
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, meta_name), 'w')
    for idx, img_name in enumerate(img_names):
        key = osp.join(lmdb_path, img_name)
        if multiprocessing_read:
            img_byte = dataset[key]
        else:
            img_path = osp.join(data_path, img_name)
            _, img_byte, _ = read_img_worker(img_path, compress_level, key)

        key_byte = key.encode('ascii')
        txn.put(key_byte, img_byte)
        # write meta information
        txt_file.write(f'{img_name}\n')
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)

        prog_bar.update()
    txn.commit()
    env.close()
    txt_file.close()
    prog_bar.close()
    print('\nFinish writing lmdb.')


def main_make_lmdb(folder_path, lmdb_path):
    img_names = sorted(list(mmcv.scandir(folder_path, recursive=False)))
    make_lmdb(data_path=folder_path, lmdb_path=lmdb_path, img_names=img_names)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--no-patch',
                        action='store_true',
                        help='whether to crop image patches')
    parser.add_argument('--no-lmdb',
                        action='store_true',
                        help='whether to generate lmdb')
    parser.add_argument('--src',
                        default='data/div2k/train',
                        help='source path')
    parser.add_argument('--tmp',
                        default='tmp/div2k_lmdb/train',
                        help='temporal path')
    parser.add_argument('--save',
                        default='data/div2k_lmdb/train.lmdb',
                        help='save path')
    parser.add_argument('--nthreads',
                        type=int,
                        default=16,
                        help='thread number for multiprocessing')
    parser.add_argument('--compression-level',
                        type=int,
                        default=0,
                        help='compression level when save png images')
    parser.add_argument('--patch-size',
                        type=int,
                        default=128,
                        help='cropped size for HR images')
    parser.add_argument('--step', type=int, default=64, help='cropping step')
    parser.add_argument('--thresh-size',
                        type=int,
                        default=0,
                        help='threshold size for HR images')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    assert not (args.no_patch
                and args.no_lmdb), 'Nothing to do; patching and LMDB are off.'
    os.makedirs(args.save)

    if args.no_patch:
        main_make_lmdb(args.src, args.save)
    else:
        if args.no_lmdb:
            main_crop_patches(args, args.src, args.save)
        else:
            os.makedirs(args.tmp)
            main_crop_patches(args, args.src, args.tmp)
            main_make_lmdb(args.tmp, args.save)
