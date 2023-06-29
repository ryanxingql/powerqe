"""Compress image datasets.

Annotation files may also created for those datasets without splits. According
to SRAnnotationDataset, each line in the annotation file contains the image
names and image shape (usually for gt), separated by a white space. For
instance: "0001_s001.png (480,480,3).

Copyright 2023 RyanXingQL

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
import multiprocessing as mp
import os
import os.path as osp

import cv2
from tqdm import tqdm


def bpg_compress(enc_cmd, dec_cmd):
    os.system(enc_cmd)
    os.system(dec_cmd)


def parse_args():
    parser = argparse.ArgumentParser(description='Compress image dataset.')
    parser.add_argument('--codec', type=str, required=True, choices=['bpg'])
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices=['div2k', 'flickr2k'])
    parser.add_argument('--max-npro', type=int, default=16)
    parser.add_argument('--quality', type=int, default=37)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.codec == 'bpg':
        BPGENC_PATH = osp.abspath('data/libbpg/bpgenc')
        BPGDEC_PATH = osp.abspath('data/libbpg/bpgdec')
        QP = args.quality

        paths = []

        if args.dataset == 'div2k':
            SRC_DIR = osp.abspath('data/div2k')
            TMP_DIR = osp.abspath(f'tmp/div2k_lq/bpg/qp{args.quality}')
            TAR_DIR = osp.abspath(f'data/div2k_lq/bpg/qp{args.quality}')

            # training set

            src_dir = osp.join(SRC_DIR, 'train')
            tmp_dir = osp.join(TMP_DIR, 'train')
            tar_dir = osp.join(TAR_DIR, 'train')
            os.makedirs(tmp_dir)
            os.makedirs(tar_dir)

            for idx in range(1, 801):
                paths.append(
                    dict(src=osp.join(src_dir, f'{idx:04d}.png'),
                         bpg=osp.join(tmp_dir, f'{idx:04d}.bpg'),
                         tar=osp.join(tar_dir, f'{idx:04d}.png')))

            # validation set

            src_dir = osp.join(SRC_DIR, 'valid')
            tmp_dir = osp.join(TMP_DIR, 'valid')
            tar_dir = osp.join(TAR_DIR, 'valid')
            os.makedirs(tmp_dir)
            os.makedirs(tar_dir)

            for idx in range(801, 901):
                paths.append(
                    dict(src=osp.join(src_dir, f'{idx:04d}.png'),
                         bpg=osp.join(tmp_dir, f'{idx:04d}.bpg'),
                         tar=osp.join(tar_dir, f'{idx:04d}.png')))

        elif args.dataset == 'flickr2k':
            SRC_DIR = osp.abspath('data/flickr2k')
            TMP_DIR = osp.abspath(f'tmp/flickr2k_lq/bpg/qp{args.quality}')
            TAR_DIR = osp.abspath(f'data/flickr2k_lq/bpg/qp{args.quality}')

            os.makedirs(TMP_DIR)
            os.makedirs(TAR_DIR)

            for idx in range(1, 2651):
                paths.append(
                    dict(src=osp.join(SRC_DIR, f'{idx:06d}.png'),
                         bpg=osp.join(TMP_DIR, f'{idx:06d}.bpg'),
                         tar=osp.join(TAR_DIR, f'{idx:06d}.png')))

            # create meta

            with open(osp.join(TAR_DIR, 'train.txt'), 'w') as file:
                for idx in tqdm(range(1, 1989), ncols=0):
                    img_name = f'{idx:06d}.png'
                    gt_path = osp.join(SRC_DIR, img_name)
                    gt = cv2.imread(gt_path)
                    h, w, c = gt.shape
                    line = f'{img_name} ({h},{w},{c})\n'
                    file.write(line)

            with open(osp.join(TAR_DIR, 'test.txt'), 'w') as file:
                for idx in tqdm(range(1989, 2651), ncols=0):
                    img_name = f'{idx:06d}.png'
                    gt_path = osp.join(SRC_DIR, img_name)
                    gt = cv2.imread(gt_path)
                    h, w, c = gt.shape
                    line = f'{img_name} ({h},{w},{c})\n'
                    file.write(line)

        # compression

        pool = mp.Pool(processes=args.max_npro)

        pbar = tqdm(total=len(paths), ncols=0)
        for path in paths:
            src_path = path['src']
            enc_cmd = f'{BPGENC_PATH} -o {path["bpg"]} -q {QP} {path["src"]}'
            dec_cmd = f'{BPGDEC_PATH} -o {path["tar"]} {path["bpg"]}'
            pool.apply_async(func=bpg_compress,
                             args=(enc_cmd, dec_cmd),
                             callback=lambda _: pbar.update())
        pool.close()
        pool.join()
        pbar.close()
