# RyanXingQL @2023
import argparse
import multiprocessing as mp
import os
from glob import glob


def hm_encode(enc_cmd):
    os.system(enc_cmd)
    return enc_cmd


def parse_args():
    parser = argparse.ArgumentParser(description='Compress video dataset.')
    parser.add_argument(
        '--max-npro',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['vimeo90k-triplet'],
    )
    parser.add_argument(
        '--src',
        type=str,
        default='tmp/planar/vimeo_triplet',
    )
    parser.add_argument(
        '--bit',
        type=str,
        default='tmp/bit/vimeo_triplet',
    )
    parser.add_argument(
        '--tar',
        type=str,
        default='tmp/comp_planar/vimeo_triplet',
    )
    parser.add_argument(
        '--hm',
        type=str,
        default='data/hm',
    )
    parser.add_argument(
        '--qp',
        type=int,
        default=37,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    pool = mp.Pool(processes=args.max_npro)

    if args.dataset == 'vimeo90k-triplet':
        vid_paths = sorted(glob(os.path.join(args.src, '*/')))
        for vid_path in vid_paths:
            vid_name = vid_path.split('/')[-2]
            bit_dir = os.path.join(args.bit, vid_name)
            tar_dir = os.path.join(args.tar, vid_name)
            os.makedirs(bit_dir)
            os.makedirs(tar_dir)
            src_paths = sorted(glob(os.path.join(vid_path, '*.yuv')))
            for src_path in src_paths:
                src_name = os.path.splitext(os.path.basename(src_path))[0]
                enc_path = os.path.join(args.hm, 'bin/TAppEncoderStatic')
                cfg_path = os.path.join(args.hm,
                                        'cfg/encoder_lowdelay_P_main.cfg')
                bit_path = os.path.join(bit_dir, src_name + '.bin')
                tar_path = os.path.join(tar_dir, src_name + '.yuv')
                log_path = os.path.join(tar_dir, src_name + '.log')
                enc_cmd = (f'{enc_path} -i {src_path} -c {cfg_path}'
                           f' -b {bit_path} -o {tar_path}'
                           f' -wdt 448 -hgt 256 -q {args.qp}'
                           ' --Level=3.1 -fr 30 -f 3'
                           f' > {log_path}')
                pool.apply_async(func=hm_encode,
                                 args=(enc_cmd, ),
                                 callback=lambda x: print(x))

    pool.close()
    pool.join()
