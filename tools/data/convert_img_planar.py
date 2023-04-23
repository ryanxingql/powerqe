# RyanXingQL @2023
import argparse
import multiprocessing as mp
import os
from glob import glob

import cv2
import numpy as np


def write_planar(img, planar_path):
    """
    img: list of (h, w) array; each list item represents a channel.
    """
    planar_file = open(planar_path, 'wb')
    for cha in img:
        h, w = cha.shape
        for ih in range(h):
            for iw in range(w):
                planar_file.write(cha[ih, iw])
    planar_file.close()


def read_planar(planar_path, fmt=((1080, 1920), (1080, 1920), (1080, 1920))):
    """
    fmt: tuple of (h, w) tuple; each tuple item represents a channel.

    https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html
    """
    planar_file = np.fromfile(planar_path, dtype=np.uint8)
    img = []
    accum = 0
    for res in fmt:
        h, w = res
        cha = planar_file[accum:(accum + h * w)].reshape(h, w)
        img.append(cha)
        accum += h * w
    return img


def vimeo90k_triplet_write_ycbcr420(seq_path, tar_path):
    ycbcr420 = []
    for idx in range(1, 4):
        img_path = os.path.join(seq_path, f'im{idx}.png')
        bgr = cv2.imread(img_path)
        h, w = bgr.shape[:2]
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        cr_sub = cv2.resize(ycrcb[..., 1], (w // 2, h // 2),
                            interpolation=cv2.INTER_AREA)
        cb_sub = cv2.resize(ycrcb[..., 2], (w // 2, h // 2),
                            interpolation=cv2.INTER_AREA)
        ycbcr420.append(ycrcb[..., 0])
        ycbcr420.append(cb_sub)  # cb before cr
        ycbcr420.append(cr_sub)
    write_planar(ycbcr420, tar_path)
    return tar_path


def vimeo90k_triplet_read_ycbcr420(src_path, tar_dir, h=256, w=448):
    ycbcr420_3frms = read_planar(
        src_path,
        fmt=(
            (h, w),
            (h // 2, w // 2),
            (h // 2, w // 2),
            (h, w),
            (h // 2, w // 2),
            (h // 2, w // 2),
            (h, w),
            (h // 2, w // 2),
            (h // 2, w // 2),
        ),
    )
    for idx in range(3):
        img_path = os.path.join(tar_dir, f'im{idx+1}.png')
        ycrcb = np.empty((h, w, 3), np.uint8)
        ycrcb[..., 0] = ycbcr420_3frms[3 * idx]
        ycrcb[..., 1] = cv2.resize(ycbcr420_3frms[3 * idx + 2], (w, h),
                                   interpolation=cv2.INTER_CUBIC)
        ycrcb[..., 2] = cv2.resize(ycbcr420_3frms[3 * idx + 1], (w, h),
                                   interpolation=cv2.INTER_CUBIC)
        bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(img_path, bgr)
    return tar_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description='Conversion between image and planar format.')
    parser.add_argument(
        '--max-npro',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['vimeo90k-triplet'],
    )
    parser.add_argument(
        '--src',
        type=str,
        default='data/vimeo_triplet/sequences',
    )
    parser.add_argument(
        '--tar',
        type=str,
        default='tmp/planar/vimeo_triplet',
    )
    parser.add_argument(
        '--revert',
        action='store_true',
    )
    parser.add_argument(
        '--src-re',
        type=str,
        default='tmp/comp_planar/vimeo_triplet',
    )
    parser.add_argument(
        '--tar-re',
        type=str,
        default='data/vimeo_triplet_lq',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    pool = mp.Pool(processes=args.max_npro)

    if args.dataset == 'vimeo90k-triplet':
        if not args.revert:
            vid_paths = sorted(glob(os.path.join(args.src, '*/')))
            for vid_path in vid_paths:
                vid_name = vid_path.split('/')[-2]
                tar_dir = os.path.join(args.tar, vid_name)
                os.makedirs(tar_dir)
                seq_paths = sorted(glob(os.path.join(vid_path, '*/')))
                for seq_path in seq_paths:
                    seq_name = seq_path.split('/')[-2]
                    tar_path = os.path.join(tar_dir, seq_name + '.yuv')
                    pool.apply_async(func=vimeo90k_triplet_write_ycbcr420,
                                     args=(
                                         seq_path,
                                         tar_path,
                                     ),
                                     callback=lambda x: print(x))
        else:
            vid_paths = sorted(glob(os.path.join(args.src_re, '*/')))
            for vid_path in vid_paths:
                vid_name = vid_path.split('/')[-2]
                vid_dir = os.path.join(args.tar_re, vid_name)
                src_paths = sorted(glob(os.path.join(vid_path, '*.yuv')))
                for src_path in src_paths:
                    src_name = os.path.splitext(os.path.basename(src_path))[0]
                    tar_dir = os.path.join(vid_dir, src_name)
                    os.makedirs(tar_dir)
                    pool.apply_async(func=vimeo90k_triplet_read_ycbcr420,
                                     args=(
                                         src_path,
                                         tar_dir,
                                     ),
                                     callback=lambda x: print(x))

    pool.close()
    pool.join()
