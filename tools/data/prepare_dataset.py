# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL @2022
import argparse
import os
import os.path as osp
from multiprocessing import Manager, Pool

import cv2
import lmdb
import mmcv
import numpy as np
from tqdm import tqdm


def crop_one_image(patch_names_all_imgs,
                   img_path,
                   crop_size,
                   step,
                   thresh_size,
                   save_folder,
                   compression_level,
                   patch_ext='.png'):
    """Crop one image into patches."""
    img_name, _ = osp.splitext(osp.basename(img_path))

    img = mmcv.imread(img_path, flag='unchanged')
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

    patch_names = []
    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            patch_name = f'{img_name}_{index:d}{patch_ext}'
            patch_names.append(patch_name)
            save_path = osp.join(save_folder, patch_name)

            patch = img[x:x + crop_size, y:y + crop_size, ...]
            cv2.imwrite(save_path, patch,
                        [cv2.IMWRITE_PNG_COMPRESSION, compression_level])

    patch_names_all_imgs += patch_names


def crop_patches(img_names,
                 nthreads,
                 compression_level,
                 crop_size,
                 step,
                 thresh_size,
                 input_folder,
                 save_folder,
                 patch_ext='.png'):
    """Crop images to patches."""
    img_paths = [osp.join(input_folder, v) for v in img_names]

    manager = Manager()
    patch_names_all_imgs = manager.list()

    prog_bar = tqdm(total=len(img_paths), ncols=0)
    pool = Pool(nthreads)
    for img_path in img_paths:
        patch_cfg = dict(img_path=img_path,
                         patch_names_all_imgs=patch_names_all_imgs,
                         crop_size=crop_size,
                         step=step,
                         thresh_size=thresh_size,
                         save_folder=save_folder,
                         compression_level=compression_level,
                         patch_ext=patch_ext)
        pool.apply_async(crop_one_image,
                         kwds=patch_cfg,
                         callback=lambda _: prog_bar.update())
    pool.close()
    pool.join()
    prog_bar.close()

    return patch_names_all_imgs


def read_img(img_path, compress_level, key):
    """Read image.

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
    return img_byte, key


def make_lmdb(data_path,
              lmdb_path,
              img_names,
              nthreads=32,
              batch=5000,
              compress_level=1,
              multiprocessing_read=False,
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
        img_names (list)
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
        multiprocessing_read (bool): Whether use multiprocessing to read all
            the images to memory. Default: False.
        nthreads (int): For multiprocessing.
    """
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("'lmdb_path' must end with '.lmdb'.")

    print(f'Source: {data_path}\nTarget: {lmdb_path}')
    print(f'Total images: {len(img_names)}')

    # read all the images to memory with multiprocessing

    if multiprocessing_read:
        dataset = {}  # use dict to keep the order for multiprocessing

        def callback(img_byte, key):
            """get the image data and update prog_bar."""
            dataset[key] = img_byte
            prog_bar.update()

        prog_bar = tqdm(total=len(img_names), ncols=0)
        pool = Pool(nthreads)
        for img_name in img_names:
            img_path = osp.join(data_path, img_name)
            key = osp.join(lmdb_path, img_name)
            pool.apply_async(read_img,
                             args=(img_path, compress_level, key),
                             callback=callback)
        pool.close()
        pool.join()
        prog_bar.close()

    # create lmdb environment

    # obtain data size by one image
    img = mmcv.imread(osp.join(data_path, img_names[0]), flag='unchanged')
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    data_size_per_img = img_byte.nbytes
    print('Data size per image is: ', data_size_per_img)

    data_size = data_size_per_img * len(img_names)
    env = lmdb.open(lmdb_path, map_size=data_size * 10)

    # (read and) write data to lmdb

    prog_bar = tqdm(total=len(img_names), ncols=0)
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, meta_name), 'w')
    for idx, img_name in enumerate(img_names):
        key = osp.join(lmdb_path, img_name)
        if multiprocessing_read:
            img_byte = dataset[key]
        else:
            img_path = osp.join(data_path, img_name)
            img_byte, _ = read_img(img_path, compress_level, key)

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
    parser.add_argument('--anno-path', default='', help='annotation path')
    parser.add_argument('--tmp',
                        default='tmp/div2k_lmdb/train',
                        help='temporal path')
    parser.add_argument('--save',
                        default='data/div2k_lmdb/train.lmdb',
                        help='save path')
    parser.add_argument('--nthreads',
                        type=int,
                        default=32,
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

    assert not (args.no_patch and args.no_lmdb), 'Nothing to do.'

    os.makedirs(args.save)

    # list all images

    if args.anno_path:
        with open(args.anno_path, 'r') as f:
            img_names = f.read().split('\n')
        img_names = [
            n.strip() for n in img_names if (n.strip() is not None and n != '')
        ]
    else:
        img_names = sorted(list(mmcv.scandir(args.src, recursive=False)))

    # generate patches and create LMDB

    if args.no_patch:
        make_lmdb(data_path=args.src,
                  lmdb_path=args.save,
                  img_names=img_names,
                  nthreads=args.nthreads)
    else:
        patch_cfg = dict(img_names=img_names,
                         nthreads=args.nthreads,
                         compression_level=args.compression_level,
                         crop_size=args.patch_size,
                         step=args.step,
                         thresh_size=args.thresh_size)

        if args.no_lmdb:
            patch_cfg['input_folder'] = args.src
            patch_cfg['save_folder'] = args.save
            crop_patches(**patch_cfg)

        else:
            patch_cfg['input_folder'] = args.src
            patch_cfg['save_folder'] = args.tmp
            os.makedirs(args.tmp)
            img_names = crop_patches(**patch_cfg)  # actually patch names

            make_lmdb(data_path=args.tmp,
                      lmdb_path=args.save,
                      img_names=img_names,
                      nthreads=args.nthreads)
