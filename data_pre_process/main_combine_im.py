import os
import sys
import os.path as op
from glob import glob as glob

cwd = op.split(op.realpath(__file__))[0]
raw_dir = op.join(cwd, '../data/div2k/raw_ds_2/')

sep_lst = [700,100,100]  # tra, val, test

type = sys.argv[1]

tar_raw_dir = op.join(cwd, '../data/div2k/raw_ds_2_combined')

if type == 'jpeg':
    cmp_dir_lst = [
        op.join(cwd, '../data/div2k/jpeg_ds_2/qf10'),
        op.join(cwd, '../data/div2k/jpeg_ds_2/qf20'),
        op.join(cwd, '../data/div2k/jpeg_ds_2/qf30'),
        op.join(cwd, '../data/div2k/jpeg_ds_2/qf40'),
        op.join(cwd, '../data/div2k/jpeg_ds_2/qf50'),
    ]
    tar_cmp_dir = op.join(cwd, '../data/div2k/jpeg_ds_2_combined')
elif type == 'hevc':
    cmp_dir_lst = [
        op.join(cwd, '../data/div2k/bpg_ds_2/qp22'),
        op.join(cwd, '../data/div2k/bpg_ds_2/qp27'),
        op.join(cwd, '../data/div2k/bpg_ds_2/qp32'),
        op.join(cwd, '../data/div2k/bpg_ds_2/qp37'),
        op.join(cwd, '../data/div2k/bpg_ds_2/qp42'),
    ]
    tar_cmp_dir = op.join(cwd, '../data/div2k/bpg_ds_2_combined')

if not op.exists(tar_raw_dir):
    os.makedirs(tar_raw_dir)
if not op.exists(tar_cmp_dir):
    os.makedirs(tar_cmp_dir)

src_raw_lst = sorted(glob(op.join(raw_dir, '*.png')))
src_num = len(src_raw_lst)
print(f'{src_num} images per subdir are found.')

tar_num = 0
acc_num = 0
for sep in sep_lst:
    tar_num += sep
    while True:
        acc_num += 1
        if acc_num > tar_num:
            break
    
        src_raw_path = src_raw_lst[acc_num-1]
        im_name = src_raw_path.split('/')[-1].split('.')[0]
        for cmp_dir in cmp_dir_lst:
            src_cmp_path = op.join(cmp_dir, im_name + '.png')
            cmp_dir_name = cmp_dir.split('/')[-1]
            
            new_path = op.join(tar_cmp_dir, f'{im_name}-{cmp_dir_name}.png')
            command_ = f'ln -s {src_cmp_path} {new_path}'
            os.system(command_)

            new_path = op.join(tar_raw_dir, f'{im_name}-{cmp_dir_name}.png')
            command_ = f'ln -s {src_raw_path} {new_path}'
            os.system(command_)
