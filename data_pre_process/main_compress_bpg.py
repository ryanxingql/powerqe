import glob
import os
import sys
import os.path as op

src_name = sys.argv[1]
libbpg_dir = sys.argv[2]
qp = sys.argv[3]  # qp, e.g., 22, 27, 32, ...
tar_name = sys.argv[4]

cwd = op.split(op.realpath(__file__))[0]
src_folder = os.path.join(cwd, f'../data/{src_name}')
tar_folder = os.path.join(cwd, f'../data/{tar_name}/qp' + qp)
bpgenc_path = os.path.join(libbpg_dir, 'bpgenc')
bpgdec_path = os.path.join(libbpg_dir, 'bpgdec')
if not os.path.exists(tar_folder):
    os.makedirs(tar_folder)

im_lst_src = sorted(glob.glob(os.path.join(src_folder, '*.png')))
num = len(im_lst_src)
for idx, im_path_src in enumerate(im_lst_src):
    im_name = im_path_src.split('/')[-1]
    im_path_tar = os.path.join(tar_folder, im_name)
    
    # m: 1, fastest but worst (big); 9, slowest but good (small); default 8
    # qp: 0-51
    # ycbcr 420p
    command_ = f'{bpgenc_path} {im_path_src} -m 1 -b 8 -q {qp} -f 420 -c ycbcr -o ./tmp.bpg && {bpgdec_path} ./tmp.bpg -b 8 -o {im_path_tar}'  # 
    os.system(command_)

    print(f'{idx + 1} / {num}: {im_path_tar}')

os.remove('./tmp.bpg')
