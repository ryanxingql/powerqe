import argparse
import math
import os
import os.path as osp
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
                        choices=[
                            'div2k', 'flickr2k', 'vimeo90k-triplet',
                            'vimeo90k-septuplet'
                        ])
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--crop-boarder', type=int, default=0)
    args = parser.parse_args()
    return args


args = parse_args()

# Collect image paths

if args.dataset == 'div2k':
    gt_dir = 'data/div2k/valid'
    lq_dir = 'data/div2k_lq/bpg/qp37/valid'
    imgs_paths = [
        dict(gt_path=osp.join(gt_dir, f'{idx:04d}.png'),
             lq_path=osp.join(lq_dir, f'{idx:04d}.png'),
             out_path=osp.join(args.out_dir, f'{idx:04d}.png'))
        for idx in range(801, 901)
    ]
if args.dataset == 'flickr2k':
    keys = []
    with open('data/flickr2k_lq/bpg/qp37/test.txt', 'r') as f:
        for line in f:
            line = line.split(' ')[0]
            if not line:
                continue
            keys.append(line)
    gt_dir = 'data/flickr2k'
    lq_dir = 'data/flickr2k_lq/bpg/qp37'
    imgs_paths = [
        dict(gt_path=osp.join(gt_dir, key),
             lq_path=osp.join(lq_dir, key),
             out_path=osp.join(args.out_dir, key)) for key in keys
    ]
if args.dataset == 'vimeo90k-triplet':
    keys = []
    with open('data/vimeo_triplet/tri_testlist.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            keys.append(line)
    gt_dir = 'data/vimeo_triplet/sequences'
    lq_dir = 'data/vimeo_triplet_lq/hm18.0/ldp/qp37'
    seqs_paths = [
        dict(gt_path=osp.join(gt_dir, key),
             lq_path=osp.join(lq_dir, key),
             out_path=osp.join(args.out_dir, key)) for key in keys
    ]
if args.dataset == 'vimeo90k-septuplet':
    keys = []
    with open('data/vimeo_septuplet/sep_testlist.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            keys.append(line)
    gt_dir = 'data/vimeo_septuplet/sequences'
    lq_dir = 'data/vimeo_septuplet_lq/hm18.0/ldp/qp37'
    seqs_paths = [
        dict(gt_path=osp.join(gt_dir, key),
             lq_path=osp.join(lq_dir, key),
             out_path=osp.join(args.out_dir, key)) for key in keys
    ]

# Calculate PSNR

if 'vimeo90k' in args.dataset:
    for tag in ['out', 'lq']:
        # Record frame-wise PSNR

        nfrms_bk = 0
        for seq_paths in tqdm(seqs_paths, ncols=0):
            # Check sequence length
            nfrms = len(glob(os.path.join(seq_paths['gt_dir'], 'im*.png')))
            if not nfrms_bk:
                assert nfrms == 3 or nfrms == 7
                nfrms_bk = nfrms
                results = dict()
                for idx in range(nfrms):
                    results[idx] = []
            assert nfrms == nfrms_bk

            # Calculate frame-wise PSNR

            for idx in range(nfrms):
                src = cv2.imread(
                    osp.join(seq_paths['gt_dir'], f'im{idx+1}.png'))
                tar = cv2.imread(
                    osp.join(seq_paths[f'{tag}_dir'], f'im{idx+1}.png'))
                psnr = cal_psnr(src, tar, crop_border=args.crop_boarder)
                results[idx].append(psnr)

        # Summarize

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
        print(f'{tag}: {ave:.4f} dB')

        for idx in range(nfrms):
            result = [r for r in results[idx] if not math.isinf(r)]
            print(f'* im{idx+1}: {np.mean(result):.4f} dB')

        if inf_frm:
            print(f'Ignored {inf_frm} frame(s) in {inf_seq} sequence(s)'
                  ' with inf PSNR.')

else:
    for tag in ['out', 'lq']:
        results = []
        for img_paths in tqdm(imgs_paths, ncols=0):
            src = cv2.imread(img_paths['gt_path'])
            tar = cv2.imread(img_paths[f'{tag}_path'])
            psnr = cal_psnr(src, tar, crop_border=args.crop_boarder)
            results.append(psnr)
        print(f'ave. PSNR ({tag}): {sum(results) / len(results):.4f} dB')
