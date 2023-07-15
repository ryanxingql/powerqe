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
    parser = argparse.ArgumentParser(description='Eval PSNR.')
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices=[
                            'div2k', 'flickr2k', 'vimeo-triplet',
                            'vimeo-septuplet', 'mfqev2'
                        ])
    parser.add_argument('--outDir', type=str, default=None)
    parser.add_argument('--crop-boarder', type=int, default=0)
    args = parser.parse_args()
    return args


def cal_imgDir_psnr(imgInfos, silent=False, crop_boarder=0):
    """[{src:imgPath, tar:imgPath}]"""
    results = []
    if not silent:
        imgInfos = tqdm(imgInfos, ncols=0)
    nIgnore = 0

    for imgInfo in imgInfos:
        src = cv2.imread(imgInfo['src'])
        tar = cv2.imread(imgInfo['tar'])
        psnr = cal_psnr(src, tar, crop_border=crop_boarder)

        if math.isinf(psnr):
            nIgnore += 1
        else:
            results.append(psnr)
    assert results, 'No valid result'
    ave_psnr = np.mean(results)
    return ave_psnr, nIgnore


def cal_lq_out_psnr(gtDir, lqDir, imgNames, args):
    if args.outDir:
        imgInfos = [
            dict(src=osp.join(gtDir, imgName),
                 tar=osp.join(args.outDir, imgName)) for imgName in imgNames
        ]
        ave_psnr, nIgnore = cal_imgDir_psnr(imgInfos=imgInfos,
                                            crop_boarder=args.crop_boarder)
        print(f'Ave. PSNR (Out): {ave_psnr:.4f} dB.')
        if nIgnore:
            print(f'{nIgnore} frames are ignored.')

    imgInfos = [
        dict(src=osp.join(gtDir, imgName), tar=osp.join(lqDir, imgName))
        for imgName in imgNames
    ]
    ave_psnr, nIgnore = cal_imgDir_psnr(imgInfos=imgInfos,
                                        crop_boarder=args.crop_boarder)
    print(f'Ave. PSNR (LQ): {ave_psnr:.4f} dB.')
    if nIgnore:
        print(f'{nIgnore} frames are ignored.')


def cal_videos_psnr(gtDir, lqDir, subDirs, args):
    """subDirs (list): list of dict with keys 'dirName' and 'imgNames'."""
    if args.outDir:
        results = []
        nIgnore_accm = 0
        for subDir in tqdm(subDirs, ncols=0):
            imgInfos = [
                dict(src=osp.join(gtDir, subDir['dirName'], imgName),
                     tar=osp.join(args.outDir, subDir['dirName'], imgName))
                for imgName in subDir['imgNames']
            ]
            psnr, nIgnore = cal_imgDir_psnr(imgInfos=imgInfos,
                                            silent=True,
                                            crop_boarder=args.crop_boarder)
            results.append(psnr)
            nIgnore_accm += nIgnore
        ave_psnr = np.mean(results)
        print(f'Ave. PSNR (Out): {ave_psnr:.4f} dB.')
        if nIgnore_accm:
            print(f'{nIgnore_accm} frames are ignored.')

    results = []
    nIgnore_accm = 0
    for subDir in tqdm(subDirs, ncols=0):
        imgInfos = [
            dict(src=osp.join(gtDir, subDir['dirName'], imgName),
                 tar=osp.join(lqDir, subDir['dirName'], imgName))
            for imgName in subDir['imgNames']
        ]
        psnr, nIgnore = cal_imgDir_psnr(imgInfos=imgInfos,
                                        silent=True,
                                        crop_boarder=args.crop_boarder)
        results.append(psnr)
        nIgnore_accm += nIgnore
    ave_psnr = np.mean(results)
    print(f'Ave. PSNR (LQ): {ave_psnr:.4f} dB.')
    if nIgnore_accm:
        print(f'{nIgnore_accm} frames are ignored.')


args = parse_args()

if args.dataset == 'div2k':
    gtDir = 'data/div2k/valid'
    lqDir = 'data/div2k_lq/bpg/qp37/valid'
    imgNames = [f'{iImg:04d}.png' for iImg in range(801, 901)]
    cal_lq_out_psnr(gtDir=gtDir, lqDir=lqDir, imgNames=imgNames, args=args)

if args.dataset == 'flickr2k':
    gtDir = 'data/flickr2k'
    lqDir = 'data/flickr2k_lq/bpg/qp37'

    imgNames = []
    with open('data/flickr2k_lq/bpg/qp37/test.txt', 'r') as f:
        for line in f:
            line = line.split(' ')[0]
            if not line:
                continue
            imgNames.append(line)

    cal_lq_out_psnr(gtDir=gtDir, lqDir=lqDir, imgNames=imgNames, args=args)

if args.dataset == 'vimeo-triplet':
    gtDir = 'data/vimeo_triplet/sequences'
    lqDir = 'data/vimeo_triplet_lq/hm18.0/ldp/qp37'
    imgNames = [f'im{iImg:d}.png' for iImg in range(1, 4)]

    subDirs = []
    with open('data/vimeo_triplet/tri_testlist.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            subDirs.append(dict(dirName=line, imgNames=imgNames))

    cal_videos_psnr(gtDir=gtDir, lqDir=lqDir, subDirs=subDirs, args=args)

if args.dataset == 'vimeo-septuplet':
    gtDir = 'data/vimeo_septuplet/sequences'
    lqDir = 'data/vimeo_septuplet_lq/hm18.0/ldp/qp37'
    imgNames = [f'im{iImg:d}.png' for iImg in range(1, 8)]

    subDirs = []
    with open('data/vimeo_septuplet/sep_testlist.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            subDirs.append(dict(dirName=line, imgNames=imgNames))

    cal_videos_psnr(gtDir=gtDir, lqDir=lqDir, subDirs=subDirs, args=args)

if args.dataset == 'mfqev2':
    gtDir = 'data/mfqev2/test'
    lqDir = 'data/mfqev2_lq/hm18.0/ldp/qp37/test'
    dirPaths = glob(osp.join(gtDir, '*/'))
    dirNames = [dirName.split('/')[-2] for dirName in dirPaths]

    subDirs = []
    for dirPath in dirPaths:
        dirName = dirPath.split('/')[-2]
        imgNames = glob(osp.join(dirPath, '*.png'))
        imgNames = [osp.basename(imgName) for imgName in imgNames]
        subDirs.append(dict(dirName=dirName, imgNames=imgNames))

    cal_videos_psnr(gtDir=gtDir, lqDir=lqDir, subDirs=subDirs, args=args)
