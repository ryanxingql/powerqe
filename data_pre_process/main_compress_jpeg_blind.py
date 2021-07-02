import os
import sys
from pathlib import Path

dataset = sys.argv[1]  # div2k
src_name = sys.argv[2]  # raw
tar_name = sys.argv[3]  # jpeg

current_dir = Path(__file__).resolve().parent

for quality in ['10', '20', '30', '40', '50']:
    tar_im_dir = (current_dir / '..' / 'data' / dataset / tar_name / ('qf' + quality)).resolve()

    if not tar_im_dir.exists():
        py_path = current_dir / 'main_compress_jpeg.py'
        os.system(f'python {py_path} {dataset} {src_name} {tar_name} {quality}')

    else:
        print(f'skip {tar_im_dir}: already exists.')
