"""
Author: RyanXingQL
"""
import argparse
import multiprocessing as mp
import os
from functools import partial
from glob import glob

import cv2
import numpy as np


def write_planar(img, planar_path):
    """Write planar.

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
    """Read planar.

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


def vimeo90k_write_ycbcr420(seq_path, tar_path, nfrms=3):
    ycbcr420 = []
    for idx in range(1, nfrms + 1):
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


def vimeo90k_read_ycbcr420(src_path, tar_dir, h=256, w=448, nfrms=3):
    ycbcr420_nfrms = read_planar(src_path,
                                 fmt=(
                                     (h, w),
                                     (h // 2, w // 2),
                                     (h // 2, w // 2),
                                 ) * nfrms)
    for idx in range(nfrms):
        img_path = os.path.join(tar_dir, f'im{idx+1}.png')
        ycrcb = np.empty((h, w, 3), np.uint8)
        ycrcb[..., 0] = ycbcr420_nfrms[3 * idx]
        ycrcb[..., 1] = cv2.resize(ycbcr420_nfrms[3 * idx + 2], (w, h),
                                   interpolation=cv2.INTER_CUBIC)
        ycrcb[..., 2] = cv2.resize(ycbcr420_nfrms[3 * idx + 1], (w, h),
                                   interpolation=cv2.INTER_CUBIC)
        bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(img_path, bgr)
    return tar_dir


def hm_encode(enc_cmd):
    os.system(enc_cmd)
    return enc_cmd


def parse_args():
    parser = argparse.ArgumentParser(description='Compress video dataset.')
    parser.add_argument('--max-nprocs', type=int, default=16)
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices=['vimeo90k-triplet', 'vimeo90k-septuplet'])
    parser.add_argument('--qp', type=int, default=37)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    HM_DIR = 'data/hm18.0'

    args = parse_args()

    if args.dataset == 'vimeo90k-triplet':
        SRC_DIR = 'data/vimeo_triplet/sequences'
        PLANAR_DIR = 'tmp/vimeo_triplet_planar'
        BIT_DIR = 'tmp/vimeo_triplet_bit/hm18.0/ldp/qp37'
        COMP_PLANAR_DIR = 'tmp/vimeo_triplet_comp_planar/hm18.0/ldp/qp37'
        TAR_DIR = 'data/vimeo_triplet_lq/hm18.0/ldp/qp37'

    elif args.dataset == 'vimeo90k-septuplet':
        SRC_DIR = 'data/vimeo_septuplet/sequences'
        PLANAR_DIR = 'tmp/vimeo_septuplet_planar'
        BIT_DIR = 'tmp/vimeo_septuplet_bit/hm18.0/ldp/qp37'
        COMP_PLANAR_DIR = 'tmp/vimeo_septuplet_comp_planar/hm18.0/ldp/qp37'
        TAR_DIR = 'data/vimeo_septuplet_lq/hm18.0/ldp/qp37'

    # img -> planar
    # according to the HM manual, HM accepts videos in raw 4:2:0
    # planar format (Y'CbCr)

    if args.dataset == 'vimeo90k-triplet':
        func_write = partial(vimeo90k_write_ycbcr420, nfrms=3)
        func_read = partial(vimeo90k_read_ycbcr420, nfrms=3)

    elif args.dataset == 'vimeo90k-septuplet':
        func_write = partial(vimeo90k_write_ycbcr420, nfrms=7)
        func_read = partial(vimeo90k_read_ycbcr420, nfrms=7)

    pool = mp.Pool(processes=args.max_nprocs)

    vid_paths = sorted(glob(os.path.join(SRC_DIR, '*/')))
    for vid_path in vid_paths:
        vid_name = vid_path.split('/')[-2]
        tar_dir = os.path.join(PLANAR_DIR, vid_name)
        os.makedirs(tar_dir)

        seq_paths = sorted(glob(os.path.join(vid_path, '*/')))
        for seq_path in seq_paths:
            seq_name = seq_path.split('/')[-2]
            tar_path = os.path.join(tar_dir, seq_name + '.yuv')
            pool.apply_async(func=func_write,
                             args=(
                                 seq_path,
                                 tar_path,
                             ),
                             callback=lambda x: print(x))

    pool.close()
    pool.join()

    # compress planar

    enc_cmd_add = f' -wdt 448 -hgt 256 -q {args.qp} --Level=3.1 -fr 30'
    if args.dataset == 'vimeo90k-triplet':
        enc_cmd_add += ' -f 3'
    elif args.dataset == 'vimeo90k-septuplet':
        enc_cmd_add += ' -f 7'

    pool = mp.Pool(processes=args.max_nprocs)

    vid_paths = sorted(glob(os.path.join(PLANAR_DIR, '*/')))
    for vid_path in vid_paths:
        vid_name = vid_path.split('/')[-2]
        bit_dir = os.path.join(BIT_DIR, vid_name)
        tar_dir = os.path.join(COMP_PLANAR_DIR, vid_name)
        os.makedirs(bit_dir)
        os.makedirs(tar_dir)

        src_paths = sorted(glob(os.path.join(vid_path, '*.yuv')))
        for src_path in src_paths:
            src_name = os.path.splitext(os.path.basename(src_path))[0]
            enc_path = os.path.join(HM_DIR, 'bin/TAppEncoderStatic')
            cfg_path = os.path.join(HM_DIR, 'cfg/encoder_lowdelay_P_main.cfg')
            bit_path = os.path.join(bit_dir, src_name + '.bin')
            tar_path = os.path.join(tar_dir, src_name + '.yuv')
            log_path = os.path.join(tar_dir, src_name + '.log')
            enc_cmd = (f'{enc_path} -i {src_path} -c {cfg_path}'
                       f' -b {bit_path} -o {tar_path}'
                       f'{enc_cmd_add}'
                       f' > {log_path}')
            pool.apply_async(func=hm_encode,
                             args=(enc_cmd, ),
                             callback=lambda x: print(x))

    pool.close()
    pool.join()

    # planar -> img

    pool = mp.Pool(processes=args.max_nprocs)

    vid_paths = sorted(glob(os.path.join(COMP_PLANAR_DIR, '*/')))
    for vid_path in vid_paths:
        vid_name = vid_path.split('/')[-2]
        vid_dir = os.path.join(TAR_DIR, vid_name)
        src_paths = sorted(glob(os.path.join(vid_path, '*.yuv')))
        for src_path in src_paths:
            src_name = os.path.splitext(os.path.basename(src_path))[0]
            tar_dir = os.path.join(vid_dir, src_name)
            os.makedirs(tar_dir)
            pool.apply_async(func=func_read,
                             args=(
                                 src_path,
                                 tar_dir,
                             ),
                             callback=lambda x: print(x))

    pool.close()
    pool.join()
