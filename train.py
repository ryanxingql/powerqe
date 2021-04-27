import math
import torch
import shutil
import dataset
import algorithm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from utils import print_n_log, get_timestr, mkdir_archived, dict2str, DistSampler, create_dataloader, arg2dict, CPUPrefetcher, init_dist, set_random_seed, Timer

def create_logger(opts_dict):
    """Make log dir -> log params -> define tensorboard writer."""
    exp_name = get_timestr() if opts_dict['algorithm']['exp_name'] == None else opts_dict['algorithm']['exp_name']
    log_dir = Path("exp") / exp_name
    # if train from scratch, or (resume training and not keep dir)
    if (not opts_dict['algorithm']['train']['load_state']['if_load']) or \
        (
            (opts_dict['algorithm']['train']['load_state']['if_load']) and \
            (not opts_dict['algorithm']['train']['load_state']['if_keep_dir'])
        ):
        mkdir_archived(log_dir)

    log_path = log_dir / "log.log"
    log_fp = open(log_path, 'a')
    msg = (
        f'> hi - {get_timestr()}\n\n'
        f'> options\n'
        f'{dict2str(opts_dict).rstrip()}'  # remove \n from dict2str()
    )
    print_n_log(msg, log_fp)

    writer = SummaryWriter(log_dir)

    ckp_save_path_pre = log_dir / 'ckp-'

    return log_fp, writer, ckp_save_path_pre

def create_data_fetcher(
        if_train=False,
        seed=None,
        num_gpu=None,
        rank=None,
        ds_type=None,
        ds_opts=None,
        enlarge_ratio=None,
        nworker_pg=None,
        bs_pg=None,
    ):
    """Define dataset -> sampler -> dataloader -> CPU datafetcher."""
    ds_cls = getattr(dataset, ds_type)
    ds = ds_cls(ds_opts)
    nsample = len(ds)
    
    if if_train:
        sampler = DistSampler(
            num_replicas=num_gpu,
            rank=rank,
            ratio=enlarge_ratio,
            ds_size=nsample,
        )
    else:
        sampler = None  # no need to sample val data
    
    loader = create_dataloader(
        if_train=if_train,
        seed=seed,
        rank=rank,
        num_worker=nworker_pg,
        batch_size=bs_pg,
        dataset=ds,
        sampler=sampler,
    )
    
    data_fetcher = CPUPrefetcher(loader)
    return data_fetcher, sampler, nsample

def cal_state(bs_pg, num_gpu, nsample, enlarge_ratio, niter, done_niter):
    bs_pe_all_gpu = bs_pg * num_gpu  # pe: per epoch
    enlarge_nsample_pe = nsample * enlarge_ratio
    niter_pe = math.ceil(enlarge_nsample_pe / bs_pe_all_gpu)  # also batch num
    nepoch = math.ceil(niter / niter_pe)
    done_nepoch = done_niter // niter_pe
    msg = (
        f'> dataloader\n'
        f'total samples: [{nsample}]\n'
        f'total niter: [{niter}]\n'
        f'total nepoch: [{nepoch}]\n'
        f'done niter: [{done_niter}]\n'
        f'done nepoch: [{done_nepoch}]'
    )
    return done_nepoch, msg

def main():
    seed = 7

    num_gpu = torch.cuda.device_count()
    opts_dict, rank = arg2dict()

    # cudnn
    if opts_dict['algorithm']['train']['if_cudnn']:
        torch.backends.cudnn.benchmark = True  # speed up
    else:
        torch.backends.cudnn.benchmark = False  # reproduce
        torch.backends.cudnn.deterministic = True  # reproduce

    # init distributed training
    init_dist(local_rank=rank, backend='nccl')

    # create logger
    if rank == 0:
        log_paras = dict(num_gpu=num_gpu)
        opts_dict.update(log_paras)
        log_fp, writer, ckp_save_path_pre = create_logger(opts_dict)

    # enlarge niter and niter_warmup
    bs_pg = opts_dict['dataset']['train']['bs_pg']
    real_bs_pg = opts_dict['algorithm']['train']['real_bs_pg']
    assert bs_pg <= real_bs_pg and real_bs_pg % bs_pg == 0, '> Check your bs and real bs!'
    inter_step = real_bs_pg // bs_pg
    opts_dict['algorithm']['train']['niter_warmup'] = opts_dict['algorithm']['train']['niter_warmup'] * inter_step if 'niter_warmup' in opts_dict['algorithm']['train'] else None  # if exists
    niter = math.ceil(opts_dict['algorithm']['train']['niter'] * inter_step)  # enlarge niter

    # set random seed for this process
    set_random_seed(seed + rank)  # if not set, seeds for numpy.random in each process are the same

    # create datafetcher
    opts_dict_ = dict(
        if_train=True,
        seed=seed,
        num_gpu=num_gpu,
        rank=rank,
        **opts_dict['dataset']['train'],
    )
    train_fetcher, train_sampler, nsample_train = create_data_fetcher(**opts_dict_)
    opts_dict_ = dict(
        if_train=False,
        **opts_dict['dataset']['val'],
    )
    val_fetcher, _, nsample_val = create_data_fetcher(**opts_dict_)

    # create algorithm
    alg_cls = getattr(algorithm, opts_dict['algorithm']['type'])
    opts_dict_ = dict(
        if_train=True,
        opts_dict=opts_dict['algorithm'],
    )
    alg = alg_cls(**opts_dict_)
    if rank == 0:
        alg.print_net(log_fp)

    # calculate epoch num
    best_val_perfrm = alg.best_val_perfrm
    done_niter = alg.done_niter
    opts_dict_ = dict(
        bs_pg=bs_pg,
        num_gpu=num_gpu,
        nsample=nsample_train,
        enlarge_ratio=opts_dict['dataset']['train']['enlarge_ratio'],
        niter=niter,
        done_niter=done_niter,
    )
    done_nepoch, msg = cal_state(**opts_dict_)
    if rank == 0:
        print_n_log(msg, log_fp)

    # create timer
    if rank == 0:
        timer = Timer()

    #torch.distributed.barrier()  # all processes wait for ending
    if rank == 0:
        msg = f'> training'
        print_n_log(msg, log_fp, if_new_line=False)

    alg.set_train_mode()
    flag_over = False
    inter_print = opts_dict['algorithm']['train']['inter_print']
    inter_val = opts_dict['algorithm']['train']['inter_val']
    if_test_baseline = opts_dict['algorithm']['train']['if_test_baseline']

    while True:
        if flag_over:
            break

        # shuffle distributed subsamplers before each epoch
        train_sampler.set_epoch(done_nepoch)

        # fetch the first batch
        train_fetcher.reset()
        train_data = train_fetcher.next()
        
        while train_data is not None:
            # validate
            if (rank == 0) and \
                (
                    (done_niter == 0 and if_test_baseline)  # (1) beginning: test baseline
                    or (done_niter == niter)  # (2) the last iter
                    or ((done_niter % inter_val == 0) and (done_niter > 0))  # (3) inter_val
                ):
                
                if done_niter == 0:  # (1)
                    _, ave_val_perfm, msg, write_dict_lst = alg.test(val_fetcher, nsample_val, mod='baseline')
                    print_n_log(msg, log_fp)
                    best_val_perfrm = dict(iter_lst=[0], perfrm=ave_val_perfm)
                else:  # (2,3)
                    if_lower, ave_val_perfm, msg2, write_dict_lst = alg.test(
                        val_fetcher, nsample_val, mod='normal'
                    )  # test

                    ckp_save_path = f'{ckp_save_path_pre}{done_niter}.pt'
                    last_ckp_save_path = f'{ckp_save_path_pre}last.pt'
                    msg = f'\n> model is saved at {ckp_save_path} and {last_ckp_save_path}.\n' + msg2

                    if_save_best = False
                    if best_val_perfrm == None:  # no pre_val
                        best_val_perfrm = dict(
                            iter_lst=[done_niter],
                            perfrm=ave_val_perfm
                        )
                        if_save_best = True
                    elif ave_val_perfm == best_val_perfrm['perfrm']:
                        best_val_perfrm['iter_lst'].append(done_niter)
                    elif (ave_val_perfm > best_val_perfrm['perfrm'] and (not if_lower)) or (ave_val_perfm < best_val_perfrm['perfrm'] and if_lower):
                        best_val_perfrm['iter_lst'] = [done_niter]
                        best_val_perfrm['perfrm'] = ave_val_perfm
                        if_save_best = True
                    msg += f"\n> best val iter lst: {best_val_perfrm['iter_lst']}; best val perfrm: {best_val_perfrm['perfrm']:.3e}"

                    alg.save_state(
                        ckp_save_path=ckp_save_path, iter=done_niter, best_val_perfrm=best_val_perfrm, if_sched=alg.if_sched
                    )  # save model
                    shutil.copy(ckp_save_path, last_ckp_save_path)  # copy as the last model
                    if if_save_best:
                        best_ckp_save_path = f'{ckp_save_path_pre}first-best.pt'
                        shutil.copy(ckp_save_path, best_ckp_save_path)  # copy as the best model

                    print_n_log(msg, log_fp)

                for write_dict in write_dict_lst:
                    writer.add_scalar(write_dict['tag'], write_dict['scalar'], done_niter)

            # show network structure
            if (rank == 0) and \
                opts_dict['algorithm']['train']['if_show_graph'] and \
                ((done_niter == inter_val) or (done_niter == alg.done_niter)):
                alg.add_graph(writer=writer, data=train_data['lq'].cuda())

            #torch.distributed.barrier()  # all processes wait for ending

            # training

            if done_niter >= niter:
                flag_over = True
                break

            alg.set_train_mode()

            flag_step = True if (done_niter + 1) % inter_step == 0 else False
            if 'additional' in opts_dict['algorithm']['train']: 
                msg, write_dict_lst, gen_im_lst = alg.update_params(
                    data=train_data,
                    iter=done_niter,
                    flag_step=flag_step,
                    inter_step=inter_step,
                    additional=opts_dict['algorithm']['train']['additional']
                )
            else:
                msg, write_dict_lst, gen_im_lst = alg.update_params(
                    data=train_data,
                    iter=done_niter,
                    flag_step=flag_step,
                    inter_step=inter_step
                )

            done_niter += 1

            #torch.distributed.barrier()  # all processes wait for ending
            if (done_niter % inter_print == 0) and (rank == 0):
                used_time = timer.get_inter()
                eta = used_time / inter_print * (niter - done_niter)
                
                msg = f'{get_timestr()}; iter: [{done_niter}]/{niter}; eta: [{eta / 3600.:.1f}] h; ' + msg
                print_n_log(msg, log_fp, if_new_line=False)
                for write_dict in write_dict_lst:
                    writer.add_scalar(write_dict['tag'], write_dict['scalar'], done_niter)
                for gen_im_item in gen_im_lst:
                    ims = gen_im_lst[gen_im_item]
                    writer.add_images(gen_im_item, ims, done_niter, dataformats='NCHW')

                timer.record()

            # fetch the next batch
            train_data = train_fetcher.next()

            # update learning rate after each iter by scheduler
            if alg.if_sched:
                alg.update_lr()
        # > end of this epoch
        done_nepoch += 1
    # > end of all epochs

    # final log
    #torch.distributed.barrier()
    if rank == 0:
        msg = (
            f'> total time: [{timer.get_total() / 3600:.1f}] h\n'
            f"> best val iter lst: {best_val_perfrm['iter_lst']}; best val perfrm: {best_val_perfrm['perfrm']}\n"
            f'> bye - {get_timestr()}'
        )
        print_n_log(msg, log_fp)

        writer.close()
        log_fp.close()

if __name__ == '__main__':
    main()
