import argparse
import os

import cv2
from mmedit.core import psnr as cal_psnr
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Eval PSNR for Vimeo-90K triplet test set.')
    parser.add_argument(
        '--gt',
        type=str,
        default='data/vimeo_triplet/sequences',
    )
    parser.add_argument(
        '--lq',
        type=str,
        default='data/vimeo_triplet_lq',
    )
    parser.add_argument(
        '--out',
        type=str,
        default='data/vimeo_triplet_lq',
    )
    args = parser.parse_args()
    return args


args = parse_args()

anno_path = os.path.join(args.gt, '../tri_testlist.txt')
keys = []
with open(anno_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # print(line)
        keys.append(line)

for tar_dir in [args.lq, args.out]:
    results = []
    results_im1 = []
    results_im2 = []
    results_im3 = []
    for key in tqdm(keys, ncols=0):
        src_seq_dir = os.path.join(args.gt, key)
        tar_seq_dir = os.path.join(tar_dir, key)

        result_seq = []
        for idx in range(1, 4):
            src_img = cv2.imread(os.path.join(src_seq_dir, f'im{idx}.png'))
            tar_img = cv2.imread(os.path.join(tar_seq_dir, f'im{idx}.png'))
            psnr = cal_psnr(src_img, tar_img)
            result_seq.append(psnr)

        results.append(sum(result_seq) / len(result_seq))
        results_im1.append(result_seq[0])
        results_im2.append(result_seq[1])
        results_im3.append(result_seq[2])

    print(f'ave. PSNR: {sum(results) / len(results)}')
    print(f'ave. PSNR for im1: {sum(results_im1) / len(results_im1)}')
    print(f'ave. PSNR for im2: {sum(results_im2) / len(results_im2)}')
    print(f'ave. PSNR for im3: {sum(results_im3) / len(results_im3)}')
