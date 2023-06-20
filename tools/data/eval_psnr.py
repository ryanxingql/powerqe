import argparse
import math
import os
from glob import glob

import cv2
import numpy as np
from mmedit.core import psnr as cal_psnr
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Eval PSNR.')
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices=['div2k', 'flickr2k', 'vimeo90k'])
    parser.add_argument('--gt', type=str, default='data/div2k/valid')
    parser.add_argument('--out',
                        type=str,
                        default='data/div2k_lq/bpg/qp37/valid')
    parser.add_argument('--if-lq', action='store_false')
    parser.add_argument('--lq',
                        type=str,
                        default='data/div2k_lq/bpg/qp37/valid')
    parser.add_argument('--anno',
                        type=str,
                        default='data/vimeo_septuplet/sep_testlist.txt')
    parser.add_argument('--crop-boarder', type=int, default=0)
    args = parser.parse_args()
    return args


args = parse_args()

tars = dict(out=args.out)
if args.if_lq:
    tars['lq'] = args.lq

if args.dataset == 'div2k':
    for tar_name, tar in tars.items():
        results = []
        for idx in tqdm(range(100), ncols=0):
            src_img = cv2.imread(os.path.join(args.gt, f'{idx+801:04d}.png'))
            tar_img = cv2.imread(os.path.join(tar, f'{idx+801:04d}.png'))
            psnr = cal_psnr(src_img, tar_img, crop_border=args.crop_boarder)
            results.append(psnr)
        print(f'ave. PSNR ({tar_name}): {sum(results) / len(results):.4f} dB')

elif args.dataset == 'vimeo90k':
    keys = []
    with open(args.anno, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # print(line)
            keys.append(line)

    for tar_name, tar in tars.items():
        # record
        nfrms_bk = 0
        for key in tqdm(keys, ncols=0):
            src_seq_dir = os.path.join(args.gt, key)
            tar_seq_dir = os.path.join(tar, key)

            # check sequence length
            nfrms = len(glob(os.path.join(src_seq_dir, 'im*.png')))
            if not nfrms_bk:
                assert nfrms == 3 or nfrms == 7
                nfrms_bk = nfrms
                results = dict()
                for idx in range(nfrms):
                    results[idx] = []
            assert nfrms == nfrms_bk

            for idx in range(nfrms):
                src_img = cv2.imread(
                    os.path.join(src_seq_dir, f'im{idx+1}.png'))
                tar_img = cv2.imread(
                    os.path.join(tar_seq_dir, f'im{idx+1}.png'))
                psnr = cal_psnr(src_img,
                                tar_img,
                                crop_border=args.crop_boarder)
                results[idx].append(psnr)

        # calculate

        inf_seq = 0
        inf_frm = 0
        ave = []
        for idx_key in range(len(keys)):
            result_ori = [results[idx][idx_key] for idx in range(nfrms)]
            result = [r for r in result_ori if not math.isinf(r)]
            if result:
                ave.append(np.mean(result))

            if len(result) < len(result_ori):
                inf_seq += 1
                inf_frm += len(result_ori) - len(result)
        ave = np.mean(ave)
        print(tar_name + f': {ave:.4f} dB')

        for idx in range(nfrms):
            result = [r for r in results[idx] if not math.isinf(r)]
            print(f'* im{idx+1}: {np.mean(result):.4f} dB')

        if inf_frm:
            print(f'(ignore {inf_frm} frame(s) in {inf_seq} sequence(s)'
                  ' with inf PSNR)')
