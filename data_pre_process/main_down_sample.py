import os
import sys
import os.path as op
from cv2 import cv2
from glob import glob

dataset = sys.argv[1]  # div2k
ratio = int(sys.argv[2])  # 2

cwd = op.split(op.realpath(__file__))[0]
im_dir = os.path.join(cwd, f'../data/{dataset}/raw')
im_ds_dir = os.path.join(cwd, f'../data/{dataset}/raw_ds_{ratio}')
if not os.path.exists(im_ds_dir):
    os.makedirs(im_ds_dir)

im_lst = sorted(glob(os.path.join(im_dir, '*.png')))
num = len(im_lst)

for idx, im_path in enumerate(im_lst):
    im = cv2.imread(im_path)
    h, w, _ = im.shape
    h_, w_ = h // ratio, w // ratio
    # note: (w, h), not (h, w)
    # inter_area mode: no Moire pattern
    im_small = cv2.resize(im, (w_, h_), interpolation=cv2.INTER_AREA)
    im_small_path = os.path.join(im_ds_dir, im_path.split('/')[-1])
    cv2.imwrite(im_small_path, im_small)
    print(f'{(idx+1)} / {num}: {im_small_path}')
