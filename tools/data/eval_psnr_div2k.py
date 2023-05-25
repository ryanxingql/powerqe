import argparse
import os

import cv2
from mmedit.core import psnr as cal_psnr
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Eval PSNR for DIV2K valid set.')
    parser.add_argument('--gt', type=str, default='data/div2k/valid/gt')
    parser.add_argument('--lq', type=str, default='data/div2k/valid/lq')
    parser.add_argument('--out', type=str, default='data/div2k/valid/lq')
    args = parser.parse_args()
    return args


args = parse_args()
for tar in [args.lq, args.out]:
    results = []
    for idx in tqdm(range(100), ncols=0):
        src_img = cv2.imread(os.path.join(args.gt, f'{idx+801:04d}.png'))
        tar_img = cv2.imread(os.path.join(tar, f'{idx+801:04d}.png'))
        psnr = cal_psnr(src_img, tar_img)
        results.append(psnr)
    print(f'ave. PSNR: {sum(results) / len(results)}')
