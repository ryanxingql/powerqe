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
    ycbcr = []
    for idx in range(1, 4):
        img_path = os.path.join(seq_path, f'im{idx}.png')
        bgr = cv2.imread(img_path)
        h, w = bgr.shape[:2]
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        cr_sub = cv2.resize(ycrcb[..., 1], (w // 2, h // 2),
                            interpolation=cv2.INTER_AREA)
        cb_sub = cv2.resize(ycrcb[..., 2], (w // 2, h // 2),
                            interpolation=cv2.INTER_AREA)
        ycbcr.append(ycrcb[..., 0])
        ycbcr.append(cb_sub)  # cb before cr
        ycbcr.append(cr_sub)
    write_planar(ycbcr, tar_path)
    return tar_path


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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    pool = mp.Pool(processes=args.max_npro)

    if args.dataset == 'vimeo90k-triplet':
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

    pool.close()
    pool.join()
