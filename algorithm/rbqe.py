import net
import torch
from cv2 import cv2
from tqdm import tqdm
from utils import BaseAlg, Timer, Recorder, tensor2im

class RBQEAlgorithm(BaseAlg):
    """use most of the BaseAlg functions."""
    def __init__(self, opts_dict, if_train):
        self.opts_dict = opts_dict
        self.if_train = if_train

        model_cls = getattr(net, 'RBQEModel')
        self.create_model(
            model_cls=model_cls,
            opts_dict=self.opts_dict['network'],
            if_train=self.if_train,
        )

        super().__init__()  # to further obtain optim, loss, etc.

    def test(
            self, test_fetcher, nsample_test, mod='normal', if_return_each=False, img_save_folder=None, if_train=True,
        ):
        """
        baseline mod: test between src and dst.
        normal mod: test between src and tar.
        if_return_each: return result of each sample.

        note: temporally support bs=1, i.e., test one by one.
        """
        self.set_eval_mode()
        msg = ''
        write_dict_lst = []
        timer = Timer()
        timer_wo_iqam = Recorder()  # for idx_out = -2

        with torch.no_grad():
            flag_save_im = True

            # assume that validation must have criterions
            if self.crit_lst is not None:
                for crit_name in self.crit_lst:
                    crit_fn = self.crit_lst[crit_name]['fn']
                    crit_unit = self.crit_lst[crit_name]['unit']
                    crit_if_focus = self.crit_lst[crit_name]['if_focus']

                    pbar = tqdm(total=nsample_test, ncols=80)
                    recorder = Recorder()
                    
                    test_fetcher.reset()
                    
                    test_data = test_fetcher.next()
                    assert len(test_data['name']) == 1, 'Only support bs=1 for test!'
                    while test_data is not None:
                        im_gt = test_data['gt'].cuda(non_blocking=True)  # assume bs=1
                        im_lq = test_data['lq'].cuda(non_blocking=True)  # assume bs=1
                        im_name = test_data['name'][0]  # assume bs=1
                        
                        if mod == 'normal':
                            if if_train:  # parse from im name
                                im_cmp_type = im_name.split('-')[-1].split('.')[0]
                                if im_cmp_type in ['qf50','qp22']:
                                    idx_out = 0
                                elif im_cmp_type in ['qf40','qp27']:
                                    idx_out = 1
                                elif im_cmp_type in ['qf30','qp32']:
                                    idx_out = 2
                                elif im_cmp_type in ['qf20','qp37']:
                                    idx_out = 3
                                elif im_cmp_type in ['qf10','qp42']:
                                    idx_out = 4

                            else:
                                idx_out = -2  # judge by IQAM
                                #idx_out = 0  # (0|1|2|3|4): for non-blind test, assign exit

                            timer.record()
                            if idx_out == -2:
                                time_wo_iqam, im_out = self.model.module_lst['net'](im_lq, idx_out=idx_out)
                                im_out.clamp_(0., 1.)  # nlevel B=1 C H W
                                timer_wo_iqam.record(time_wo_iqam)
                            else:
                                im_out = self.model.module_lst['net'](im_lq, idx_out=idx_out).clamp_(0., 1.)  # nlevel B=1 C H W
                            timer.record_inter()

                            perfm = crit_fn(
                                torch.squeeze(im_out, 0), torch.squeeze(im_gt, 0)
                            )
                            
                            if flag_save_im and (img_save_folder is not None):  # save im
                                im = tensor2im(torch.squeeze(im_out, 0))
                                save_path = img_save_folder / (str(im_name) + '.png')
                                cv2.imwrite(str(save_path), im)
                        
                        elif mod == 'baseline':
                            timer.record()
                            perfm = crit_fn(
                                torch.squeeze(im_lq, 0), torch.squeeze(im_gt, 0)
                            )
                            timer.record_inter()
                        recorder.record(perfm)
                        
                        _msg = f'{im_name}: [{perfm:.3e}] {crit_unit:s}'
                        if if_return_each:
                            msg += _msg + '\n'
                        pbar.set_description(_msg)
                        pbar.update()
                        
                        test_data = test_fetcher.next()

                    flag_save_im = False
                    
                    # cal ave
                    ave_perfm = recorder.get_ave()
                    write_dict_lst.append(
                        dict(
                            tag=f'{crit_name} (val)',
                            scalar=ave_perfm,
                        )
                    )  # only for validation during training
                    pbar.close()
                    if mod == 'normal':
                        if idx_out == -2:
                            ave_time_wo_iqam = timer_wo_iqam.get_ave()
                            fps_wo_iqam = 1. / ave_time_wo_iqam
                            msg += f'fps without IQAM: {fps_wo_iqam:.1f}; fps with Python-based IQAM is much slower than the official implementation of MATLAB-based IQAM.\n'
                        msg += f'> {crit_name}: [{ave_perfm:.3e}] {crit_unit}\n'
                    elif mod == 'baseline':
                        msg += f'> baseline {crit_name}: [{ave_perfm:.3e}] {crit_unit}\n'
                    
                    if crit_if_focus:
                        report_perfrm = ave_perfm
                        if_lower = crit_fn.if_lower
                
                if if_train:  # validation
                    return if_lower, report_perfrm, msg.rstrip(), write_dict_lst
                else:  # test
                    return msg.rstrip(), timer.get_ave_inter()
        
            else:  # only get tar (available only for test)
                pbar = tqdm(total=nsample_test, ncols=80)
                test_fetcher.reset()
                test_data = test_fetcher.next()
                assert len(test_data['name']) == 1, 'Only support bs=1 for test!'

                while test_data is not None:
                    im_lq = test_data['lq'].cuda(non_blocking=True)  # assume bs=1
                    im_name = test_data['name'][0]  # assume bs=1

                    timer.record()
                    im_out = self.model.module_lst['net'](im_lq).clamp_(0., 1.)
                    timer.record_inter()

                    if img_save_folder is not None:  # save im
                        im = tensor2im(torch.squeeze(im_out, 0))
                        save_path = img_save_folder / (str(im_name) + '.png')
                        cv2.imwrite(str(save_path), im)
                    
                    pbar.update()
                    test_data = test_fetcher.next()

                pbar.close()
                msg += f'> no ground-truth data; test done.\n'

                return msg.rstrip(), timer.get_ave_inter()

    def update_net_params(
            self,
            data,
            flag_step,
            inter_step,
            additional
        ):
        """available for simple loss func. for complex loss such as relativeganloss, please write your own func."""
        data_lq = data['lq'].cuda(non_blocking=True)
        data_gt = data['gt'].cuda(non_blocking=True)

        data_out_lst = self.model.module_lst['net'](data_lq)  # nlevel B C H W
        nl, nb = data_out_lst.shape[0:2]

        # select the images from the last output as demos
        self.gen_im_lst = dict(
            data_lq=data['lq'][:3],
            data_gt=data['gt'][:3],
            generated=data_out_lst[-1].detach()[:3].cpu().clamp_(0., 1.),
        )  # for torch.utils.tensorboard.writer.SummaryWriter.add_images: NCHW tensor is ok

        data_name_lst = data['name']
        loss_total = 0
        for loss_item in self.loss_lst.keys():
            loss_dict = self.loss_lst[loss_item]

            loss_unweighted = 0
            for idx_data in range(nb):
                cmp_type = data_name_lst[idx_data].split('-')[-1].split('.')[0]
                loss_weight_lst = additional['weight_out'][cmp_type]

                for idx_l in range(nl):
                    opts_dict_ = dict(
                        inp=data_out_lst[idx_l, idx_data, ...],
                        ref=data_gt[idx_data, ...],
                    )
                    loss_unweighted += loss_weight_lst[idx_l] * loss_dict['fn'](**opts_dict_)
            
            loss_unweighted /= float(nb)
            setattr(self, loss_item, loss_unweighted.item())  # for recorder

            loss_ = loss_dict['weight'] * loss_unweighted
            loss_total += loss_

        loss_total /= float(inter_step)  # multiple backwards and step once, thus mean
        loss_total.backward()  # must backward only once; otherwise the graph is freed after the first backward
        setattr(self, 'net_loss', loss_total.item())  # for recorder

        setattr(self, 'net_lr', self.optim_lst['net'].param_groups[0]['lr'])  # for recorder
        if flag_step:
            self.optim_lst['net'].step()
            self.optim_lst['net'].zero_grad()
