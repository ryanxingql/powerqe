import os
import sys
from pathlib import Path

dataset = sys.argv[1]  # div2k
src_name = sys.argv[2]  # raw
tar_name = sys.argv[3]  # bpg
qp = sys.argv[4]  # 37
libbpg_dir = sys.argv[5]  # /xxx/libbpg-0.9.8

current_dir = Path(__file__).resolve().parent
src_im_dir = (current_dir / '..' / 'data' / dataset / src_name).resolve()
tar_im_dir = (current_dir / '..' / 'data' / dataset / tar_name / ('qp' + qp)).resolve()
bpgenc_path = Path(libbpg_dir) / 'bpgenc'
bpgdec_path = Path(libbpg_dir) / 'bpgdec'
if not tar_im_dir.exists():
    tar_im_dir.mkdir(parents=True)

src_im_lst = sorted(src_im_dir.glob('*.png'))
tmp_path = current_dir / 'tmp.bpg'

num = len(src_im_lst)
for idx, im_path_src in enumerate(src_im_lst):
    im_name = im_path_src.name
    im_path_tar = tar_im_dir / im_name
    
    # m: 1, fastest but worst (big); 9, slowest but good (small); default 8
    # qp: 0-51
    # ycbcr 420p

    command_ = (
        f'{bpgenc_path} {im_path_src} -m 1 -b 8 -q {qp} -f 420 -c ycbcr -o {tmp_path} && '
        f'{bpgdec_path} {tmp_path} -b 8 -o {im_path_tar}'
    )
    os.system(command_)

    print(f'{idx + 1} / {num}: {im_path_tar}')

os.remove(tmp_path)
