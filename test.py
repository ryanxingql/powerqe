import torch
import dataset
import algorithm
from pathlib import Path
from utils import Recorder, arg2dict, init_dist, get_timestr, dict2str, print_n_log, create_dataloader, CPUPrefetcher, Timer

def create_logger(opts_dict):
    """Make log dir -> log params."""
    exp_name = get_timestr() if opts_dict['algorithm']['exp_name'] == None else opts_dict['algorithm']['exp_name']
    log_dir = Path("exp") / exp_name
    img_save_folder = log_dir / 'test_im_enhanced'
    if opts_dict['algorithm']['test']['if_save_im'] and not (img_save_folder.exists()):
        img_save_folder.mkdir(parents=True)

    log_path = log_dir / "log-val.log"
    log_fp = open(log_path, 'w')
    msg = (
        f'> hi - {get_timestr()}\n\n'
        f'> options\n'
        f'{dict2str(opts_dict).rstrip()}'  # remove \n from dict2str()
        )
    print_n_log(msg, log_fp)

    return log_fp, img_save_folder

def create_data_fetcher(
        ds_type=None,
        ds_opts=None,
        ):
    """Define dataset -> dataloader -> CPU datafetcher."""
    ds_cls = getattr(dataset, ds_type)
    ds = ds_cls(ds_opts)
    nsample = len(ds)
    
    loader = create_dataloader(
        if_train=False,
        dataset=ds,
        )
    
    data_fetcher = CPUPrefetcher(loader)

    return data_fetcher, nsample

def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    opts_dict, _ = arg2dict()

    init_dist(local_rank=0, backend='nccl')

    log_fp, img_save_folder = create_logger(opts_dict)

    # datafetcher
    test_fetcher, nsample_test = create_data_fetcher(**opts_dict['dataset']['test'])

    # create algorithm
    alg_cls = getattr(algorithm, opts_dict['algorithm']['type'])
    opts_dict_ = dict(
        if_train=False,
        opts_dict=opts_dict['algorithm'],
        )
    alg = alg_cls(**opts_dict_)
    
    alg.print_net(log_fp)  # print para message

    msg = f'> testing'
    print_n_log(msg, log_fp, if_new_line=False)

    timer = Timer()

    # test baseline criterion
    if opts_dict['algorithm']['test']['criterion'] is not None:
        msg, ave_spf = alg.test(test_fetcher, nsample_test, mod='baseline', if_return_each=False, if_train=False)
        print_n_log(msg, log_fp)

    # test
    msg, ave_spf = alg.test(
        test_fetcher, nsample_test, mod='normal', if_return_each=False, img_save_folder=img_save_folder, if_train=False
    )
    ave_fps = 1. / ave_spf
    msg += (
        f'\nave. fps by model: [{ave_fps:.1f}]\n' 
        f'\n> total time: [{timer.get_total() / 3600:.1f}] h\n'
        f'> bye - {get_timestr()}'
    )
    print_n_log(msg, log_fp)

    log_fp.close()

if __name__ == '__main__':
    main()
