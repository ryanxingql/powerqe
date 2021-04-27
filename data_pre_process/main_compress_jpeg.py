import os
import sys
import numpy as np
import os.path as op
from cv2 import cv2
from glob import glob

src_name = sys.argv[1]
quality = sys.argv[2]  # qf, e.g., 10, 20, 30, ...
tar_name = sys.argv[3]

cwd = op.split(op.realpath(__file__))[0]
src_im_lst = sorted(glob(os.path.join(cwd, f'../data/{src_name}/*.png')))
tar_folder = os.path.join(cwd, f'../data/{tar_name}/qf' + quality)
if not os.path.exists(tar_folder):
  os.makedirs(tar_folder)

tot = len(src_im_lst)
count = 0
for src_im_path in src_im_lst:
    src_im = cv2.imread(src_im_path)
    src_im_name = src_im_path.split('/')[-1]
    im_new_path = os.path.join(tar_folder,  src_im_name)

    count += 1
    print(f'{count} / {tot}: {im_new_path}')

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    encimg = cv2.imencode('.jpg', src_im, encode_param)[1].tobytes()  # bytes class
    nparr = np.frombuffer(encimg, np.byte)
    img2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite(im_new_path, img2)
