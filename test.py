from pathlib import Path

import torch

import dataset
import algorithm
from utils import create_logger, arg2dict, dict2str, create_dataloader, CPUPrefetcher, CUDATimer


def mkdir_and_create_logger(opts_dict, rank):
    exp_name = opts_dict['algorithm']['exp_name']
    log_dir = Path("exp") / exp_name

    img_save_folder = log_dir / 'enhanced_images'
    if opts_dict['algorithm']['test']['if_save_im'] and not (img_save_folder.exists()):
        img_save_folder.mkdir(parents=True)

    log_path = log_dir / "log_test.log"
    logger = create_logger(log_path, rank=rank, mode='w')

    return img_save_folder, logger


def create_data_fetcher(ds_type=None, ds_opts=None):
    """Define data-set, data-loader and CPU-based data-fetcher."""
    ds_cls = getattr(dataset, ds_type)
    ds = ds_cls(ds_opts)
    num_samples = len(ds)
    loader = create_dataloader(if_train=False, dataset=ds)
    data_fetcher = CPUPrefetcher(loader)
    return num_samples, data_fetcher


def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    opts_dict, _ = arg2dict()

    img_save_folder, logger = mkdir_and_create_logger(opts_dict, rank=0)

    # record hyper-params

    msg = f'hyper parameters\n{dict2str(opts_dict).rstrip()}'  # remove \n from dict2str()
    logger.info(msg)

    # create data-fetcher

    num_samples_test, test_fetcher = create_data_fetcher(**opts_dict['dataset']['test'])

    # create algorithm

    alg_cls = getattr(algorithm, opts_dict['algorithm']['name'])
    opts_dict_ = dict(if_train=False, if_dist=False, opts_dict=opts_dict['algorithm'])
    alg = alg_cls(**opts_dict_)
    alg.model.print_module(logger)

    timer = CUDATimer()
    timer.start_record()

    if_return_each = opts_dict['algorithm']['test']['if_return_each']

    # test baseline: dst vs. src

    if opts_dict['algorithm']['test']['criterion'] is not None:
        msg = alg.test(test_fetcher, num_samples_test, if_baseline=True, if_return_each=if_return_each, if_train=False)
        logger.info(msg)

    # test: tar vs. src

    msg = alg.test(test_fetcher, num_samples_test, if_baseline=False, if_return_each=if_return_each,
                   img_save_folder=img_save_folder, if_train=False)

    total_time = timer.get_inter() / 3600.
    msg += f'\ntotal time: [{total_time:.1f}] h'
    logger.info(msg)


if __name__ == '__main__':
    main()
