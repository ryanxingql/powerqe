import os
import sys

from pathlib import Path

dataset = sys.argv[1]  # div2k
src_name = sys.argv[2]  # raw
dst_name = sys.argv[3]  # jpeg

current_dir = Path(__file__).resolve().parent
src_dir = (current_dir / '..' / 'data' / dataset / src_name).resolve()
comb_src_dir = (current_dir / '..' / 'data' / dataset / (src_name + '_combined')).resolve()

if not comb_src_dir.exists():
    comb_src_dir.mkdir()

if dst_name == 'jpeg':
    dst_dir_lst = [
        (current_dir / '..' / 'data' / dataset / dst_name / 'qf10').resolve(),
        (current_dir / '..' / 'data' / dataset / dst_name / 'qf20').resolve(),
        (current_dir / '..' / 'data' / dataset / dst_name / 'qf30').resolve(),
        (current_dir / '..' / 'data' / dataset / dst_name / 'qf40').resolve(),
        (current_dir / '..' / 'data' / dataset / dst_name / 'qf50').resolve()
    ]

elif dst_name == 'bpg':
    dst_dir_lst = [
        (current_dir / '..' / 'data' / dataset / dst_name / 'qp22').resolve(),
        (current_dir / '..' / 'data' / dataset / dst_name / 'qp27').resolve(),
        (current_dir / '..' / 'data' / dataset / dst_name / 'qp32').resolve(),
        (current_dir / '..' / 'data' / dataset / dst_name / 'qp37').resolve(),
        (current_dir / '..' / 'data' / dataset / dst_name / 'qp42').resolve()
    ]

comb_dst_dir = (current_dir / '..' / 'data' / dataset / (dst_name + '_combined')).resolve()

if not comb_dst_dir.exists():
    comb_dst_dir.mkdir()

src_lst = sorted(src_dir.glob('*.png'))
src_num = len(src_lst)
print(f'{src_num} ref images are found.')

for src_path in src_lst:
    im_name = src_path.stem

    for dst_dir in dst_dir_lst:
        dst_path = dst_dir / (str(im_name) + '.png')
        dst_type = dst_dir.stem

        new_name = f'{str(im_name)}_{str(dst_type)}.png'

        new_path = comb_src_dir / new_name
        command_ = f'ln -s {src_path} {new_path}'
        os.system(command_)

        print(command_)

        new_path = comb_dst_dir / new_name
        command_ = f'ln -s {dst_path} {new_path}'
        os.system(command_)

        print(command_)
