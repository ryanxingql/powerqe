# RyanXingQL @2023
import argparse
import multiprocessing as mp
import os

from tqdm import tqdm


def bpg_compress(enc_cmd, dec_cmd):
    os.system(enc_cmd)
    os.system(dec_cmd)


def parse_args():
    parser = argparse.ArgumentParser(description='Compress image dataset.')
    parser.add_argument(
        '--max-npro',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['div2k'],
    )
    parser.add_argument(
        '--root',
        type=str,
        default='data/div2k',
    )
    parser.add_argument(
        '--tmp',
        type=str,
        default='tmp/bpg/div2k',
    )
    parser.add_argument(
        '--codec',
        type=str,
        required=True,
        choices=['bpg'],
    )
    parser.add_argument(
        '--libbpg',
        type=str,
        default='data/libbpg',
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=37,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    MAX_NPRO = args.max_npro
    QP = args.quality
    if args.codec == 'bpg':
        BPGENC_PATH = os.path.abspath(os.path.join(args.libbpg, 'bpgenc'))
        BPGDEC_PATH = os.path.abspath(os.path.join(args.libbpg, 'bpgdec'))

    if args.dataset == 'div2k':
        # training set

        pool = mp.Pool(processes=MAX_NPRO)

        SRC_DIR = os.path.abspath(os.path.join(args.root, 'train/gt'))
        TMP_DIR = os.path.abspath(os.path.join(args.tmp, 'train/lq/bpg/qp37'))
        TAR_DIR = os.path.abspath(os.path.join(args.root, 'train/lq/bpg/qp37'))
        os.makedirs(TMP_DIR)
        os.makedirs(TAR_DIR)

        pbar = tqdm(total=800, ncols=0)
        for idx in range(1, 801):
            src_path = os.path.join(SRC_DIR, f'{idx:04d}.png')
            bpg_path = os.path.join(TMP_DIR, f'{idx:04d}.bpg')
            tar_path = os.path.join(TAR_DIR, f'{idx:04d}.png')
            enc_cmd = f'{BPGENC_PATH} -o {bpg_path} -q {QP} {src_path}'
            dec_cmd = f'{BPGDEC_PATH} -o {tar_path} {bpg_path}'
            pool.apply_async(func=bpg_compress,
                             args=(
                                 enc_cmd,
                                 dec_cmd,
                             ),
                             callback=lambda _: pbar.update())
        pool.close()
        pool.join()
        pbar.close()

        # validation set

        pool = mp.Pool(processes=MAX_NPRO)

        SRC_DIR = os.path.abspath(os.path.join(args.root, 'valid/gt'))
        TMP_DIR = os.path.abspath(os.path.join(args.tmp, 'valid/lq/bpg/qp37'))
        TAR_DIR = os.path.abspath(os.path.join(args.root, 'valid/lq/bpg/qp37'))
        os.makedirs(TMP_DIR)
        os.makedirs(TAR_DIR)

        pbar = tqdm(total=100, ncols=0)
        for idx in range(801, 901):
            src_path = os.path.join(SRC_DIR, f'{idx:04d}.png')
            bpg_path = os.path.join(TMP_DIR, f'{idx:04d}.bpg')
            tar_path = os.path.join(TAR_DIR, f'{idx:04d}.png')
            enc_cmd = f'{BPGENC_PATH} -o {bpg_path} -q {QP} {src_path}'
            dec_cmd = f'{BPGDEC_PATH} -o {tar_path} {bpg_path}'
            pool.apply_async(func=bpg_compress,
                             args=(
                                 enc_cmd,
                                 dec_cmd,
                             ),
                             callback=lambda _: pbar.update())
        pool.close()
        pool.join()
        pbar.close()
