import os
import sys
from pathlib import Path

dataset = sys.argv[1]  # div2k
src_name = sys.argv[2]  # raw
tar_name = sys.argv[3]  # bpg
libbpg_dir = sys.argv[4]  # /xxx/libbpg-0.9.8

current_dir = Path(__file__).resolve().parent

for quality in ['42', '37', '32', '27', '22']:
    tar_im_dir = (current_dir / '..' / 'data' / dataset / tar_name / ('qp' + quality)).resolve()

    if not tar_im_dir.exists():
        py_path = current_dir / 'main_compress_bpg.py'
        os.system(f'python {py_path} {dataset} {src_name} {tar_name} {quality} {libbpg_dir}')
