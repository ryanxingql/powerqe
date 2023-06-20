# RyanXingQL @2023
import argparse
import multiprocessing as mp
import os
import os.path as osp

from tqdm import tqdm


def bpg_compress(enc_cmd, dec_cmd):
    os.system(enc_cmd)
    os.system(dec_cmd)


def parse_args():
    parser = argparse.ArgumentParser(description='Compress image dataset.')
    parser.add_argument('--codec', type=str, required=True, choices=['bpg'])
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices=['div2k', 'flickr2k'])
    parser.add_argument('--max-npro', type=int, default=16)
    parser.add_argument('--quality', type=int, default=37)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.codec == 'bpg':
        BPGENC_PATH = osp.abspath('data/libbpg/bpgenc')
        BPGDEC_PATH = osp.abspath('data/libbpg/bpgdec')
        QP = args.quality

        if args.dataset == 'div2k':
            SRC_DIR = osp.abspath('data/div2k')
            TMP_DIR = osp.abspath(f'tmp/div2k_lq/bpg/qp{args.quality}')
            TAR_DIR = osp.abspath(f'data/div2k_lq/bpg/qp{args.quality}')

            # training set

            pool = mp.Pool(processes=args.max_npro)

            src_dir = osp.join(SRC_DIR, 'train')
            tmp_dir = osp.join(TMP_DIR, 'train')
            tar_dir = osp.join(TAR_DIR, 'train')
            os.makedirs(tmp_dir)
            os.makedirs(tar_dir)

            pbar = tqdm(total=800, ncols=0)
            for idx in range(1, 801):
                src_path = osp.join(src_dir, f'{idx:04d}.png')
                bpg_path = osp.join(tmp_dir, f'{idx:04d}.bpg')
                tar_path = osp.join(tar_dir, f'{idx:04d}.png')
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

            pool = mp.Pool(processes=args.max_npro)

            src_dir = osp.join(SRC_DIR, 'valid')
            tmp_dir = osp.join(TMP_DIR, 'valid')
            tar_dir = osp.join(TAR_DIR, 'valid')
            os.makedirs(tmp_dir)
            os.makedirs(tar_dir)

            pbar = tqdm(total=100, ncols=0)
            for idx in range(801, 901):
                src_path = osp.join(src_dir, f'{idx:04d}.png')
                bpg_path = osp.join(tmp_dir, f'{idx:04d}.bpg')
                tar_path = osp.join(tar_dir, f'{idx:04d}.png')
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

        elif args.dataset == 'flickr2k':
            SRC_DIR = osp.abspath('data/flickr2k')
            TMP_DIR = osp.abspath(f'tmp/flickr2k_lq/bpg/qp{args.quality}')
            TAR_DIR = osp.abspath(f'data/flickr2k_lq/bpg/qp{args.quality}')

            os.makedirs(TMP_DIR)
            os.makedirs(TAR_DIR)

            # create meta

            with open(osp.join(TAR_DIR, 'train.txt'), 'w') as file:
                for num in range(1, 1989):
                    line = str(num).zfill(6) + '.png\n'
                    file.write(line)

            with open(osp.join(TAR_DIR, 'test.txt'), 'w') as file:
                for num in range(1989, 2651):
                    line = str(num).zfill(6) + '.png\n'
                    file.write(line)

            # compress

            pool = mp.Pool(processes=args.max_npro)
            pbar = tqdm(total=2650, ncols=0)
            for idx in range(1, 2651):
                src_path = osp.join(SRC_DIR, f'{idx:06d}.png')
                bpg_path = osp.join(TMP_DIR, f'{idx:06d}.bpg')
                tar_path = osp.join(TAR_DIR, f'{idx:06d}.png')
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
