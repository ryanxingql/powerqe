import argparse
import os
from glob import glob

import cv2
from mmedit.core import psnr as cal_psnr
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Eval PSNR for Vimeo-90K septuplet test set.')
    parser.add_argument('--anno',
                        type=str,
                        default='data/vimeo_septuplet/sep_testlist.txt')
    parser.add_argument('--gt',
                        type=str,
                        default='data/vimeo_septuplet/sequences')
    parser.add_argument('--lq', type=str, default='data/vimeo_septuplet_lq')
    parser.add_argument('--out', type=str, default='data/vimeo_septuplet_lq')
    args = parser.parse_args()
    return args


args = parse_args()

keys = []
with open(args.anno, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # print(line)
        keys.append(line)

for tar_dir in [args.lq, args.out]:
    nfrms_bk = -1
    result_per_frame_all_seqs = []
    for key in tqdm(keys, ncols=0):
        src_seq_dir = os.path.join(args.gt, key)
        tar_seq_dir = os.path.join(tar_dir, key)

        nfrms = len(glob(os.path.join(src_seq_dir, 'im*.png')))
        if nfrms_bk == -1:
            assert nfrms == 3 or nfrms == 7
            nfrms_bk = nfrms
        assert nfrms == nfrms_bk

        result_per_frame = []
        for idx in range(nfrms):
            src_img = cv2.imread(os.path.join(src_seq_dir, f'im{idx+1}.png'))
            tar_img = cv2.imread(os.path.join(tar_seq_dir, f'im{idx+1}.png'))
            psnr = cal_psnr(src_img, tar_img)
            result_per_frame.append(psnr)

        if not result_per_frame_all_seqs:
            result_per_frame_all_seqs = result_per_frame
        else:
            result_per_frame_all_seqs = [
                i + j
                for i, j in zip(result_per_frame_all_seqs, result_per_frame)
            ]
    result_per_frame_all_seqs = [
        r / len(keys) for r in result_per_frame_all_seqs
    ]
    print(f'{sum(result_per_frame_all_seqs) / nfrms:.4f}')
    for idx in range(nfrms):
        print(f'idx {idx}: {result_per_frame_all_seqs[idx]:.4f}')
