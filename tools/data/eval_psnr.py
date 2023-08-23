"""Copyright 2023 RyanXingQL.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at
https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import math
import os.path as osp
from glob import glob

import cv2
import numpy as np
from mmedit.core import psnr as cal_psnr
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Eval PSNR.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["div2k", "flickr2k", "vimeo-triplet", "vimeo-septuplet", "mfqev2"],
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--crop-boarder", type=int, default=0)
    args = parser.parse_args()
    return args


def cal_imgdir_psnr(img_infos, silent=False, crop_boarder=0):
    """[{src:img path, tar:img path}]"""
    results = []
    if not silent:
        img_infos = tqdm(img_infos, ncols=0)
    n_ignore = 0

    for img_info in img_infos:
        src = cv2.imread(img_info["src"])
        tar = cv2.imread(img_info["tar"])
        psnr = cal_psnr(src, tar, crop_border=crop_boarder)

        if math.isinf(psnr):
            n_ignore += 1
        else:
            results.append(psnr)
    assert results, "No valid result"
    ave_psnr = np.mean(results)
    return ave_psnr, n_ignore


def cal_lq_out_psnr(gt_dir, lq_dir, img_names, args):
    if args.out_dir:
        img_infos = [
            dict(src=osp.join(gt_dir, img_name), tar=osp.join(args.out_dir, img_name))
            for img_name in img_names
        ]
        ave_psnr, n_ignore = cal_imgdir_psnr(
            img_infos=img_infos, crop_boarder=args.crop_boarder
        )
        print(f"Ave. PSNR (Out): {ave_psnr:.4f} dB.")
        if n_ignore:
            print(f"{n_ignore} frames are ignored.")

    img_infos = [
        dict(src=osp.join(gt_dir, img_name), tar=osp.join(lq_dir, img_name))
        for img_name in img_names
    ]
    ave_psnr, n_ignore = cal_imgdir_psnr(
        img_infos=img_infos, crop_boarder=args.crop_boarder
    )
    print(f"Ave. PSNR (LQ): {ave_psnr:.4f} dB.")
    if n_ignore:
        print(f"{n_ignore} frames are ignored.")


def cal_videos_psnr(gt_dir, lq_dir, sub_dirs, args):
    """sub_dirs (list): list of dict with keys 'dir_name' and 'img_names'."""
    if args.out_dir:
        results = []
        n_ignore_accm = 0
        for sub_dir in tqdm(sub_dirs, ncols=0):
            img_infos = [
                dict(
                    src=osp.join(gt_dir, sub_dir["dir_name"], img_name),
                    tar=osp.join(args.out_dir, sub_dir["dir_name"], img_name),
                )
                for img_name in sub_dir["img_names"]
            ]
            psnr, n_ignore = cal_imgdir_psnr(
                img_infos=img_infos, silent=True, crop_boarder=args.crop_boarder
            )
            results.append(psnr)
            n_ignore_accm += n_ignore
        ave_psnr = np.mean(results)
        print(f"Ave. PSNR (Out): {ave_psnr:.4f} dB.")
        if n_ignore_accm:
            print(f"{n_ignore_accm} frames are ignored.")

    results = []
    n_ignore_accm = 0
    for sub_dir in tqdm(sub_dirs, ncols=0):
        img_infos = [
            dict(
                src=osp.join(gt_dir, sub_dir["dir_name"], img_name),
                tar=osp.join(lq_dir, sub_dir["dir_name"], img_name),
            )
            for img_name in sub_dir["img_names"]
        ]
        psnr, n_ignore = cal_imgdir_psnr(
            img_infos=img_infos, silent=True, crop_boarder=args.crop_boarder
        )
        results.append(psnr)
        n_ignore_accm += n_ignore
    ave_psnr = np.mean(results)
    print(f"Ave. PSNR (LQ): {ave_psnr:.4f} dB.")
    if n_ignore_accm:
        print(f"{n_ignore_accm} frames are ignored.")


args = parse_args()

if args.dataset == "div2k":
    gt_dir = "data/div2k/valid"
    lq_dir = "data/div2k_lq/bpg/qp37/valid"
    img_names = [f"{idx_img:04d}.png" for idx_img in range(801, 901)]
    cal_lq_out_psnr(gt_dir=gt_dir, lq_dir=lq_dir, img_names=img_names, args=args)

if args.dataset == "flickr2k":
    gt_dir = "data/flickr2k"
    lq_dir = "data/flickr2k_lq/bpg/qp37"

    img_names = []
    with open("data/flickr2k_lq/bpg/qp37/test.txt", "r") as f:
        for line in f:
            line = line.split(" ")[0]
            if not line:
                continue
            img_names.append(line)

    cal_lq_out_psnr(gt_dir=gt_dir, lq_dir=lq_dir, img_names=img_names, args=args)

if args.dataset == "vimeo-triplet":
    gt_dir = "data/vimeo_triplet/sequences"
    lq_dir = "data/vimeo_triplet_lq/hm18.0/ldp/qp37"
    img_names = [f"im{idx_img:d}.png" for idx_img in range(1, 4)]

    sub_dirs = []
    with open("data/vimeo_triplet/tri_testlist.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sub_dirs.append(dict(dir_name=line, img_names=img_names))

    cal_videos_psnr(gt_dir=gt_dir, lq_dir=lq_dir, sub_dirs=sub_dirs, args=args)

if args.dataset == "vimeo-septuplet":
    gt_dir = "data/vimeo_septuplet/sequences"
    lq_dir = "data/vimeo_septuplet_lq/hm18.0/ldp/qp37"
    img_names = [f"im{idx_img:d}.png" for idx_img in range(1, 8)]

    sub_dirs = []
    with open("data/vimeo_septuplet/sep_testlist.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sub_dirs.append(dict(dir_name=line, img_names=img_names))

    cal_videos_psnr(gt_dir=gt_dir, lq_dir=lq_dir, sub_dirs=sub_dirs, args=args)

if args.dataset == "mfqev2":
    gt_dir = "data/mfqev2/test"
    lq_dir = "data/mfqev2_lq/hm18.0/ldp/qp37/test"
    dir_paths = glob(osp.join(gt_dir, "*/"))
    dir_names = [dir_name.split("/")[-2] for dir_name in dir_paths]

    sub_dirs = []
    for dir_path in dir_paths:
        dir_name = dir_path.split("/")[-2]
        img_names = glob(osp.join(dir_path, "*.png"))
        img_names = [osp.basename(img_name) for img_name in img_names]
        sub_dirs.append(dict(dir_name=dir_name, img_names=img_names))

    cal_videos_psnr(gt_dir=gt_dir, lq_dir=lq_dir, sub_dirs=sub_dirs, args=args)
