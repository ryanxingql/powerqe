import net
import torch
from cv2 import cv2
from tqdm import tqdm

from utils import BaseAlg, CUDATimer, Recorder, tensor2im


class RBQEAlgorithm(BaseAlg):
    def __init__(self, opts_dict, if_train, if_dist):
        model_cls = getattr(net, 'RBQEModel')  # !!!
        super().__init__(opts_dict=opts_dict, model_cls=model_cls, if_train=if_train, if_dist=if_dist)

    def accum_gradient(self, module, stage, group, data, inter_step, additional):
        data_lq = data['lq'].cuda(non_blocking=True)
        data_gt = data['gt'].cuda(non_blocking=True)
        data_out_lst = module(inp_t=data_lq, if_train=True)

        nl, nb = data_out_lst.shape[0:2]  # nlevel, batch size

        num_show = 3
        self._im_lst = dict(
            data_lq=data['lq'][:num_show],
            data_gt=data['gt'][:num_show],
            generated=data_out_lst[-1].detach()[:num_show].cpu().clamp_(0., 1.),
        )  # show images from the last exit

        loss_total = torch.tensor(0., device="cuda")
        for loss_name in self.loss_lst[stage][group]:
            loss_dict = self.loss_lst[stage][group][loss_name]

            loss_unweighted = 0.
            for idx_data in range(nb):
                im_type = data['name'][idx_data].split('_')[-1].split('.')[0]
                loss_weight_lst = additional['weight_out'][im_type]

                for idx_level in range(nl):
                    opts_dict_ = dict(
                        inp=data_out_lst[idx_level, idx_data, ...],
                        ref=data_gt[idx_data, ...],
                    )
                    loss_unweighted += loss_weight_lst[idx_level] * loss_dict['fn'](**opts_dict_)
            loss_unweighted /= float(nb)

            setattr(self, f'{loss_name}_{group}', loss_unweighted.item())  # for recorder

            loss_ = loss_dict['weight'] * loss_unweighted
            loss_total += loss_

        loss_total /= float(inter_step)  # multiple backwards and step once, thus mean
        loss_total.backward()  # must backward only once; otherwise the graph is freed after the first backward
        setattr(self, f'loss_{group}', loss_total.item())  # for recorder

    @torch.no_grad()
    def test(self, data_fetcher, num_samples, if_baseline=False, if_return_each=False, img_save_folder=None,
             if_train=True):
        """
        val (in training): idx_out=0/1/2/3/4
        test: idx_out=-2, record time wo. iqa
        """
        if if_baseline or if_train:
            assert self.crit_lst is not None, 'NO METRICS!'

        if self.crit_lst is not None:
            if_tar_only = False
            msg = 'dst vs. src | ' if if_baseline else 'tar vs. src | '
        else:
            if_tar_only = True
            msg = 'only get dst | '

        report_dict = None

        recorder_dict = dict()
        for crit_name in self.crit_lst:
            recorder_dict[crit_name] = Recorder()

        write_dict_lst = []
        timer = CUDATimer()

        # validation baseline: no iqa, no parse name
        # validation, not baseline: no iqa, parse name
        # test baseline: no iqa, no parse name
        # test, no baseline, iqa, no parse name
        if_iqa = True if (not if_train) and (not if_baseline) else False
        if if_iqa:
            timer_wo_iqam = Recorder()
            idx_out = -2  # testing; judge by IQAM
        if_parse_name = True if if_train and (not if_baseline) else False

        self.set_eval_mode()

        data_fetcher.reset()
        test_data = data_fetcher.next()
        assert len(test_data['name']) == 1, 'ONLY SUPPORT bs==1!'

        pbar = tqdm(total=num_samples, ncols=100)

        while test_data is not None:
            im_lq = test_data['lq'].cuda(non_blocking=True)  # assume bs=1
            im_name = test_data['name'][0]  # assume bs=1

            if if_parse_name:
                im_type = im_name.split('_')[-1].split('.')[0]
                if im_type in ['qf50', 'qp22']:
                    idx_out = 0
                elif im_type in ['qf40', 'qp27']:
                    idx_out = 1
                elif im_type in ['qf30', 'qp32']:
                    idx_out = 2
                elif im_type in ['qf20', 'qp37']:
                    idx_out = 3
                elif im_type in ['qf10', 'qp42']:
                    idx_out = 4
                else:
                    raise Exception(f"im_type IS {im_type}, NO MATCHING TYPE!")

            timer.start_record()
            if if_tar_only:
                if if_iqa:
                    time_wo_iqa, im_out = self.model.net[self.model.infer_subnet](inp_t=im_lq, idx_out=idx_out).clamp_(0., 1.)
                else:
                    im_out = self.model.net[self.model.infer_subnet](inp_t=im_lq, idx_out=idx_out).clamp_(0., 1.)
                timer.record_inter()
            else:
                im_gt = test_data['gt'].cuda(non_blocking=True)  # assume bs=1
                if if_baseline:
                    im_out = im_lq
                else:
                    if if_iqa:
                        time_wo_iqa, im_out = self.model.net[self.model.infer_subnet](inp_t=im_lq, idx_out=idx_out)
                        im_out = im_out.clamp_(0., 1.)
                    else:
                        im_out = self.model.net[self.model.infer_subnet](inp_t=im_lq, idx_out=idx_out).clamp_(0., 1.)
                timer.record_inter()

                _msg = f'{im_name} | '

                for crit_name in self.crit_lst:
                    crit_fn = self.crit_lst[crit_name]['fn']
                    crit_unit = self.crit_lst[crit_name]['unit']

                    perfm = crit_fn(torch.squeeze(im_out, 0), torch.squeeze(im_gt, 0))
                    recorder_dict[crit_name].record(perfm)

                    _msg += f'[{perfm:.3e}] {crit_unit:s} | '

                _msg = _msg[:-3]
                if if_return_each:
                    msg += _msg + '\n'
                pbar.set_description(_msg)

            if if_iqa:
                timer_wo_iqam.record(time_wo_iqa)

            if img_save_folder is not None:  # save im
                im = tensor2im(torch.squeeze(im_out, 0))
                save_path = img_save_folder / (str(im_name) + '.png')
                cv2.imwrite(str(save_path), im)

            pbar.update()
            test_data = data_fetcher.next()
        pbar.close()

        if not if_tar_only:
            for crit_name in self.crit_lst:
                crit_unit = self.crit_lst[crit_name]['unit']
                crit_if_focus = self.crit_lst[crit_name]['if_focus']

                ave_perfm = recorder_dict[crit_name].get_ave()
                msg += f'{crit_name} | [{ave_perfm:.3e}] {crit_unit} | '

                write_dict_lst.append(dict(tag=f'{crit_name} (val)', scalar=ave_perfm))

                if crit_if_focus:
                    report_dict = dict(ave_perfm=ave_perfm, lsb=self.crit_lst[crit_name]['fn'].lsb)

        ave_fps = 1. / timer.get_ave_inter()
        msg += f'ave. fps | [{ave_fps:.1f}]'

        if if_iqa:
            ave_time_wo_iqam = timer_wo_iqam.get_ave()
            fps_wo_iqam = 1. / ave_time_wo_iqam
            msg += f' | ave. fps wo. IQAM | [{fps_wo_iqam:.1f}]'

        if if_train:
            assert report_dict is not None
            return msg.rstrip(), write_dict_lst, report_dict
        else:
            return msg.rstrip()
