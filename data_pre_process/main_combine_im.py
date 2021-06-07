import os
import sys

from pathlib import Path

dataset = sys.argv[1]  # div2k
ref_name = sys.argv[2]  # raw
src_name = sys.argv[3]  # jpeg

current_dir = Path(__file__).resolve().parent
ref_dir = (current_dir / '..' / 'data' / dataset / ref_name).resolve()
comb_ref_dir = (current_dir / '..' / 'data' / dataset / (ref_name + '_combined')).resolve()

if not comb_ref_dir.exists():
    comb_ref_dir.mkdir()

if src_name == 'jpeg':
    src_dir_lst = [
        (current_dir / '..' / 'data' / dataset / src_name / 'qf10').resolve(),
        (current_dir / '..' / 'data' / dataset / src_name / 'qf20').resolve(),
        (current_dir / '..' / 'data' / dataset / src_name / 'qf30').resolve(),
        (current_dir / '..' / 'data' / dataset / src_name / 'qf40').resolve(),
        (current_dir / '..' / 'data' / dataset / src_name / 'qf50').resolve()
    ]

elif src_name == 'bpg':
    src_dir_lst = [
        (current_dir / '..' / 'data' / dataset / src_name / 'qp22').resolve(),
        (current_dir / '..' / 'data' / dataset / src_name / 'qp27').resolve(),
        (current_dir / '..' / 'data' / dataset / src_name / 'qp32').resolve(),
        (current_dir / '..' / 'data' / dataset / src_name / 'qp37').resolve(),
        (current_dir / '..' / 'data' / dataset / src_name / 'qp42').resolve()
    ]

comb_src_dir = (current_dir / '..' / 'data' / dataset / (src_name + '_combined')).resolve()

if not comb_src_dir.exists():
    comb_src_dir.mkdir()

ref_lst = sorted(ref_dir.glob('*.png'))
ref_num = len(ref_lst)
print(f'{ref_num} ref images are found.')

for ref_path in ref_lst:
    im_name = ref_path.stem

    for src_dir in src_dir_lst:
        src_path = src_dir / (str(im_name) + '.png')
        src_type = src_dir.stem

        new_name = f'{str(im_name)}_{str(src_type)}.png'

        new_path = comb_ref_dir / new_name
        command_ = f'ln -s {ref_path} {new_path}'
        os.system(command_)

        print(command_)

        new_path = comb_src_dir / new_name
        command_ = f'ln -s {src_path} {new_path}'
        os.system(command_)

        print(command_)
