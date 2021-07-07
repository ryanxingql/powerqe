import math
import shutil
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import dataset
import algorithm
from utils import mkdir_archived, dict2str, DistSampler, create_dataloader, arg2dict, \
    CPUPrefetcher, init_dist, set_random_seed, CUDATimer, create_logger


def mkdir_and_create_logger(opts_dict, if_del_arc=False, rank=0):
    """Make log dir (also used when testing) and tensorboard writer."""
    exp_name = opts_dict['algorithm']['exp_name']
    log_dir = Path("exp") / exp_name

    if_load_ = False
    if_warn_ = False
    if opts_dict['algorithm']['train']['load_state']['if_load']:
        ckp_load_path = opts_dict['algorithm']['train']['load_state']['opts']['ckp_load_path']
        if ckp_load_path is None:
            ckp_load_path = Path('exp') / opts_dict['algorithm']['exp_name'] / 'ckp_last.pt'

        if ckp_load_path.exists():
            if_load_ = True
        else:
            if_warn_ = True

    if not if_load_:
        if_mkdir_ = True
    else:
        if opts_dict['algorithm']['train']['load_state']['opts']['if_keep_dir']:
            if_mkdir_ = False
        else:
            if_mkdir_ = True

    if if_mkdir_ and rank == 0:
        mkdir_archived(log_dir, if_del_arc=if_del_arc)

    log_path = log_dir / "log_train.log"
    logger = create_logger(log_path, rank=rank, mode='a')

    tb_writer = SummaryWriter(log_dir) if rank == 0 else None

    ckp_save_path_pre = log_dir / 'ckp_'
    return logger, tb_writer, ckp_save_path_pre, if_warn_


def create_data_fetcher(if_train=False, seed=None, num_gpu=None, rank=None, ds_type=None, ds_opts=None,
                        enlarge_ratio=None, nworker_pg=None, bs_pg=None):
    """Define data-set, data-sampler, data-loader and CPU-based data-fetcher."""
    ds_cls = getattr(dataset, ds_type)
    ds = ds_cls(ds_opts)
    num_samples = len(ds)

    sampler = DistSampler(num_replicas=num_gpu, rank=rank, ratio=enlarge_ratio, ds_size=num_samples) if if_train \
        else None

    loader = create_dataloader(if_train=if_train, seed=seed, rank=rank, num_worker=nworker_pg, batch_size=bs_pg,
                               dataset=ds, sampler=sampler)

    data_fetcher = CPUPrefetcher(loader)
    return num_samples, sampler, data_fetcher


def cal_state(batch_size_per_gpu, num_gpus, num_samples, enlarge_ratio, num_iters, done_num_iters):
    bs_per_epoch_all_gpu = batch_size_per_gpu * num_gpus
    enlarge_num_samples_pe = num_samples * enlarge_ratio
    niter_per_epoch = math.ceil(enlarge_num_samples_pe / bs_per_epoch_all_gpu)  # also batch num
    num_epochs = math.ceil(num_iters / niter_per_epoch)
    done_num_epochs = done_num_iters // niter_per_epoch
    done_iter_this_epoch = done_num_iters % niter_per_epoch
    msg = (
        f'data-loader for training\n'
        f'[{num_samples}] training samples in total.\n'
        f'[{num_epochs}] epochs in total, [{done_num_epochs}] epochs finished.\n'
        f'[{num_iters}] iterations in total, [{done_num_iters}] iterations finished.'
    )
    return done_iter_this_epoch, done_num_epochs, msg


def main():
    opts_dict, opts_aux_dict = arg2dict()

    torch.backends.cudnn.benchmark = True if opts_dict['algorithm']['train']['if_cudnn'] else False
    torch.backends.cudnn.deterministic = True if not opts_dict['algorithm']['train']['if_cudnn'] else False

    num_gpu = torch.cuda.device_count()
    log_paras = dict(num_gpu=num_gpu)
    opts_dict.update(log_paras)

    if_dist = True if num_gpu > 1 else False
    rank = opts_aux_dict['rank']
    if if_dist:
        init_dist(local_rank=rank, backend='nccl')

    # Create logger

    if_del_arc = opts_aux_dict['if_del_arc']
    logger, tb_writer, ckp_save_path_pre, if_warn_ = mkdir_and_create_logger(opts_dict, if_del_arc=if_del_arc,
                                                                             rank=rank)

    if if_warn_:
        logger.info('if_load is True, but NO PRE-TRAINED MODEL!')

    # Record hyper-params

    msg = opts_aux_dict['note']
    msg += f'\nhyper parameters\n{dict2str(opts_dict).rstrip()}'  # remove \n from dict2str()
    logger.info(msg)

    # Enlarge niter

    bs_pg = opts_dict['dataset']['train']['bs_pg']
    real_bs_pg = opts_dict['algorithm']['train']['real_bs_pg']
    assert bs_pg <= real_bs_pg and real_bs_pg % bs_pg == 0, 'CHECK bs AND real bs!'
    inter_step = real_bs_pg // bs_pg

    opts_ = opts_dict['algorithm']['train']['niter']
    niter_lst = list(map(int, opts_['niter']))
    niter_name_lst = opts_['name']
    if_75 = True if ('if_75' in opts_) and opts_['if_75'] else False
    num_stage = len(niter_lst)
    end_niter_lst = [sum(niter_lst[:is_]) for is_ in range(1, num_stage + 1)]
    niter = end_niter_lst[-1]  # all stages
    niter = math.ceil(niter * inter_step)  # enlarge niter

    # Set random seed for this process

    seed = opts_dict['algorithm']['train']['seed']
    set_random_seed(seed + rank)  # if not set, seeds for numpy.random in each process are the same

    # Create data-fetcher

    opts_dict_ = dict(if_train=True, seed=seed, num_gpu=num_gpu, rank=rank, **opts_dict['dataset']['train'])
    num_samples_train, train_sampler, train_fetcher = create_data_fetcher(**opts_dict_)
    opts_dict_ = dict(if_train=False, **opts_dict['dataset']['val'])
    num_samples_val, _, val_fetcher = create_data_fetcher(**opts_dict_)

    # Create algorithm

    alg_cls = getattr(algorithm, opts_dict['algorithm']['name'])
    opts_dict_ = dict(opts_dict=opts_dict['algorithm'], if_train=True, if_dist=if_dist)
    alg = alg_cls(**opts_dict_)
    alg.model.print_module(logger)

    # Calculate epoch num

    enlarge_ratio = opts_dict['dataset']['train']['enlarge_ratio']
    done_niter = alg.done_niter
    opts_dict_ = dict(batch_size_per_gpu=bs_pg, num_gpus=num_gpu, num_samples=num_samples_train,
                      enlarge_ratio=enlarge_ratio, num_iters=niter, done_num_iters=done_niter)
    done_iter_this_epoch, done_num_epochs, msg = cal_state(**opts_dict_)
    logger.info(msg)

    # Create timer

    timer = CUDATimer()
    timer.start_record()

    # Train

    best_val_perfrm = alg.best_val_perfrm

    inter_print = opts_dict['algorithm']['train']['inter_print']
    inter_val = opts_dict['algorithm']['train']['inter_val']
    if_test_baseline = opts_dict['algorithm']['train']['if_test_baseline']
    additional = opts_dict['algorithm']['train']['additional'] if 'additional' in \
                                                                  opts_dict['algorithm']['train'] else dict()

    alg.set_train_mode()
    if_all_over = False
    if_val_end_of_stage = False  # invalid at the start of training
    while True:
        if if_all_over:
            break  # leave the training process

        train_sampler.set_epoch(done_num_epochs)  # shuffle distributed sub-samplers before each epoch
        train_fetcher.reset()
        train_data = train_fetcher.next()  # fetch the first batch

        while train_data is not None:
            # Validate
            # done_niter == alg.done_niter == 0, if_test_baseline == True: test baseline (to be recorded at tb as step 0)
            # done_niter == alg.done_niter != 0, if_keep_dir == False: test baseline and val (to be recorded at tb as step alg.done_niter)
            # done_niter != alg.done_niter, done_niter % inter_val == 0: val
            # done_niter != alg.done_niter, if_val_end_of_stage == True: val
            _if_test_baseline = False
            _if_val = False
            if done_niter == alg.done_niter:
                if alg.done_niter == 0 and if_test_baseline:
                    _if_test_baseline = True
                elif alg.done_niter != 0 and not opts_dict['algorithm']['train']['load_state']['opts']['if_keep_dir']:
                    _if_test_baseline = True
                    _if_val = True
            else:
                if (done_niter % inter_val == 0) or if_val_end_of_stage:
                    _if_val = True

            if rank == 0 and (_if_test_baseline or _if_val):
                if _if_test_baseline:
                    msg, tb_write_dict_lst, report_dict = alg.test(val_fetcher, num_samples_val, if_baseline=True)
                    logger.info(msg)
                    if done_niter == 0:
                        best_val_perfrm = dict(iter_lst=[0], perfrm=report_dict['ave_perfm'])

                    for tb_write_dict in tb_write_dict_lst:
                        tb_writer.add_scalar(tb_write_dict['tag'], tb_write_dict['scalar'], global_step=0)

                if _if_val:
                    msg, tb_write_dict_lst, report_dict = alg.test(val_fetcher, num_samples_val, if_baseline=False)

                    ckp_save_path = f'{ckp_save_path_pre}{done_niter}.pt'
                    last_ckp_save_path = f'{ckp_save_path_pre}last.pt'
                    msg = f'model is saved at [{ckp_save_path}] and [{last_ckp_save_path}].\n' + msg

                    perfrm = report_dict['ave_perfm']
                    lsb = report_dict['lsb']
                    if best_val_perfrm is None:  # no pre_val
                        best_val_perfrm = dict(iter_lst=[done_niter], perfrm=perfrm)
                        if_save_best = True
                    elif perfrm == best_val_perfrm['perfrm']:
                        best_val_perfrm['iter_lst'].append(done_niter)
                        if_save_best = False
                    else:
                        if_save_best = False
                        if (not lsb) and (perfrm > best_val_perfrm['perfrm']):
                            if_save_best = True
                            best_val_perfrm = dict(iter_lst=[done_niter], perfrm=perfrm)
                        elif lsb and (perfrm < best_val_perfrm['perfrm']):
                            if_save_best = True
                            best_val_perfrm = dict(iter_lst=[done_niter], perfrm=perfrm)
                    msg += f"\nbest iterations: [{best_val_perfrm['iter_lst']}]" \
                           f" | validation performance: [{best_val_perfrm['perfrm']:.3e}]"

                    alg.save_state(
                        ckp_save_path=ckp_save_path, idx_iter=done_niter, best_val_perfrm=best_val_perfrm,
                        if_sched=alg.if_sched
                    )  # save model
                    shutil.copy(ckp_save_path, last_ckp_save_path)  # copy as the last model
                    if if_save_best:
                        best_ckp_save_path = f'{ckp_save_path_pre}first_best.pt'
                        shutil.copy(ckp_save_path, best_ckp_save_path)  # copy as the best model

                    logger.info(msg)

                    for tb_write_dict in tb_write_dict_lst:
                        tb_writer.add_scalar(tb_write_dict['tag'], tb_write_dict['scalar'], done_niter)

            # Show network structure

            if (rank == 0) and \
                    opts_dict['algorithm']['train']['if_show_graph'] and \
                    ((done_niter == inter_val) or (done_niter == alg.done_niter)):
                alg.add_graph(writer=tb_writer, data=train_data['lq'].cuda())

            # Determine whether to exit or not

            if done_niter >= niter:
                if not if_75:  # done_niter == niter
                    if_all_over = True  # no more training after the upper validation
                    break  # leave the training data fetcher, but still in the training-validation loop
                else:
                    if _if_val and (best_val_perfrm['iter_lst'][0] / done_niter <= 0.75):
                        if_all_over = True
                        print('qualified for exit.')
                        break

            # Figure out the current stage

            if_val_end_of_stage = False
            if done_niter < niter:
                stage_now = niter_name_lst[0]
                end_niter_this_stage = 0
                for is_, end_niter in enumerate(end_niter_lst):
                    if done_niter < end_niter:
                        stage_now = niter_name_lst[is_]
                        end_niter_this_stage = end_niter
                        if_val_end_of_stage = True if done_niter == (end_niter - 1) else False
                        break
            else:
                stage_now = niter_name_lst[-1]  # keep using the optim/scheduler of the last stage when training

            # Train one batch/iteration

            alg.set_train_mode()

            if_step = True if (done_niter + 1) % inter_step == 0 else False
            msg, tb_write_dict_lst, im_lst = alg.update_params(
                stage=stage_now,
                data=train_data,
                if_step=if_step,
                inter_step=inter_step,
                additional=additional,
            )

            done_niter += 1

            # Record & Display

            if done_niter % inter_print == 0 or if_val_end_of_stage:
                used_time = timer.record_and_get_inter()
                et = timer.get_sum_inter() / 3600
                timer.start_record()

                if done_niter < niter:
                    eta = used_time / inter_print * (niter - done_niter) / 3600
                    msg = (f'{stage_now} | iter [{done_niter}]/{end_niter_this_stage}/{niter} | '
                           f'eta/et: [{eta:.1f}]/{et:.1f} h | ' + msg)
                else:
                    msg = (f'automatically stop if qualified | iter [{done_niter}]/{niter} | ' + msg)
                logger.info(msg)

                if rank == 0:
                    for tb_write_dict in tb_write_dict_lst:
                        tb_writer.add_scalar(tb_write_dict['tag'], tb_write_dict['scalar'], done_niter)
                    for im_item in im_lst:
                        ims = im_lst[im_item]
                        tb_writer.add_images(im_item, ims, done_niter, dataformats='NCHW')

            train_data = train_fetcher.next()  # fetch the next batch

        # end of this epoch

        done_num_epochs += 1

    # end of all epochs

    if rank == 0:  # only rank0 conduct tests and record the best_val_perfrm
        timer.record_inter()
        tot_time = timer.get_sum_inter() / 3600
        msg = (
            f"best iterations: [{best_val_perfrm['iter_lst']}] | validation performance: [{best_val_perfrm['perfrm']:.3e}]\n"
            f'total time: [{tot_time:.1f}] h'
        )
        logger.info(msg)


if __name__ == '__main__':
    main()
