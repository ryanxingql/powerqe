import torch
from tqdm import tqdm
from cv2 import cv2
import net
from utils import BaseAlg, Recorder, return_optimizer, tensor2im, Timer

class ESRGANAlgorithm(BaseAlg):
    def __init__(self, if_train, opts_dict):
        #super().__init__() # totally different from the BaseAlg; no need to/cannot super

        self.if_train = if_train
        self.opts_dict = opts_dict

        model_cls = getattr(net, 'ESRGANModel')
        self.create_model(
            model_cls=model_cls,
            if_train=self.if_train,
            opts_dict=self.opts_dict['network']
            )

        if self.if_train:
            self.if_sched = self.opts_dict['train']['scheduler'].pop('if_sched')

            self.create_loss_func(opts_dict=self.opts_dict['train']['loss'], if_use_cuda=True)

            params_lst = dict(
                gen=self.model.module_lst['gen'].parameters(),
                dis=self.model.module_lst['dis'].parameters(),
                warmup=self.model.module_lst['gen'].parameters(),
                )
            self.create_optimizer(params_lst=params_lst, opts_dict=self.opts_dict['train']['optimizer'])

            if self.if_sched:
                optim_lst = dict(
                    gen=self.optim_lst['gen'],
                    dis=self.optim_lst['dis'],
                    )
                self.create_scheduler(optim_lst=optim_lst, opts_dict=self.opts_dict['train']['scheduler'])

            self.create_criterion(opts_dict=self.opts_dict['val']['criterion'])
        else:
            if self.opts_dict['test']['criterion'] is not None:
                self.create_criterion(opts_dict=self.opts_dict['test']['criterion'])
            else:
                self.crit_lst = None

        if not self.if_train:  # load g for test
            load_item_lst = ['module_gen']
            self.done_niter = self.load_state(ckp_load_path=self.opts_dict['test']['ckp_load_path'], load_item_lst=load_item_lst, if_dist=True)
        else:
            if self.opts_dict['train']['load_state']['if_load']:  # load g w./wo. d for training
                load_item_lst = ['module_gen', 'module_dis', 'optim_gen', 'optim_dis']
                if self.if_sched:
                    load_item_lst += ['sched_gen', 'sched_dis']
                self.done_niter, self.best_val_perfrm = self.load_state(
                    ckp_load_path=self.opts_dict['train']['load_state']['ckp_load_path'],
                    load_item_lst=load_item_lst,
                    if_dist=True,
                )
            else:  # train from scratch
                self.done_niter = 0
                self.best_val_perfrm = None

    def add_graph(self, writer, data):
        self.set_eval_mode()
        writer.add_graph(self.model.module_lst['gen'].module, data)

    def create_optimizer(self, params_lst, opts_dict):
        """
        in addition to the BaseAlg func, optim-warmup is used for training G only.
        """
        self.optim_lst = dict()

        opts_ttur = opts_dict.pop('TTUR')
        if_ttur = False
        if opts_ttur['if_ttur']:
            if_ttur = True
            lr = opts_ttur['lr']

        opts_warmup = opts_dict.pop('warmup')
        opts_dict_ = dict(
            name=opts_warmup['type'],
            params=params_lst['warmup'],
            opts=opts_warmup['opts'],
            )
        optim = return_optimizer(**opts_dict_)
        self.optim_lst['warmup'] = optim
        
        for optim_item in opts_dict:
            if if_ttur:
                if optim_item == 'gen':
                    new_lr = lr / 2.
                elif optim_item == 'dis':
                    new_lr = lr * 2.
                opts_dict[optim_item]['opts']['lr'] = new_lr
            
            opts_dict_ = dict(
                name=opts_dict[optim_item]['type'],
                params=params_lst[optim_item],
                opts=opts_dict[optim_item]['opts'],
                )
            optim = return_optimizer(**opts_dict_)
            self.optim_lst[optim_item] = optim

    def test(
            self, test_fetcher, nsample_test, mod='normal', if_return_each=False, img_save_folder=None, if_train=True,
        ):
        """
        the same as that in BaseAlg, except net to gen
        """
        self.set_eval_mode()
        msg = ''
        write_dict_lst = []
        timer = Timer()

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
                            timer.record()
                            im_out = self.model.module_lst['gen'](im_lq).clamp_(0., 1.)
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
                    im_out = self.model.module_lst['gen'](im_lq).clamp_(0., 1.)
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
        
    def update_dis_params(self, data, flag_step, inter_step):
        data_lq = data['lq'].cuda(non_blocking=True)
        data_gt = data['gt'].cuda(non_blocking=True)
        data_out = self.model.module_lst['gen'](data_lq)

        opts_dict_ = dict(
            dis=self.model.module_lst['dis'],
            data_real=data_gt,
            data_fake=data_out,
            inter_step=inter_step,
            mode='dis',
            )
        loss_real_item, loss_fake_item = self.loss_lst['RelativisticGANLoss']['fn'](**opts_dict_)
        setattr(self, 'dis_real_loss', loss_real_item)  # for recorder
        setattr(self, 'dis_fake_loss', loss_fake_item)  # for recorder

        setattr(self, 'dis_lr', self.optim_lst['dis'].param_groups[0]['lr'])  # for recorder
        if flag_step:
            self.optim_lst['dis'].step()
            self.optim_lst['dis'].zero_grad()

    def update_gen_params(self, data, flag_step, inter_step, if_warmup=False):
        data_lq = data['lq'].cuda(non_blocking=True)
        data_gt = data['gt'].cuda(non_blocking=True)
        data_out = self.model.module_lst['gen'](data_lq)

        self.gen_im_lst = dict(
            data_lq=data['lq'][:3],
            data_gt=data['gt'][:3],
            generated=data_out.detach()[:3].cpu().clamp_(0., 1.),
            )  # for torch.utils.tensorboard.writer.SummaryWriter.add_images: NCHW tensor is ok

        loss_dict = self.loss_lst['CharbonnierLoss']
        opts_dict_ = dict(
            inp=data_out,
            ref=data_gt,
            )
        loss_l1_unweighted = loss_dict['fn'](**opts_dict_)
        setattr(self, 'CharbonnierLoss', loss_l1_unweighted.item())  # for recorder
        loss_l1 = loss_dict['weight'] * loss_l1_unweighted

        if not if_warmup:
            loss_dict = self.loss_lst['VGGLoss']
            opts_dict_ = dict(
                inp=data_out,
                ref=data_gt,
                )
            loss_unweighted = loss_dict['fn'](**opts_dict_)
            setattr(self, 'VGGLoss', loss_unweighted.item())  # for recorder
            loss_vgg = loss_dict['weight'] * loss_unweighted

            loss_dict = self.loss_lst['RelativisticGANLoss']
            opts_dict_ = dict(
                dis=self.model.module_lst['dis'],
                data_real=data_gt,
                data_fake=data_out,
                inter_step=inter_step,
                mode='gen',
                )
            loss_unweighted = loss_dict['fn'](**opts_dict_)
            setattr(self, 'RelativisticGANLoss', loss_unweighted.item())  # for recorder
            loss_rgan = loss_dict['weight'] * loss_unweighted

            loss_total = loss_l1 + loss_vgg + loss_rgan
        else:
            loss_total = loss_l1_unweighted

        loss_total /= float(inter_step)  # multiple backwards and step once, thus mean
        loss_total.backward()  # must backward only once; otherwise the graph is freed after the first backward
        setattr(self, 'gen_loss', loss_total.item())  # for recorder

        if if_warmup:
            setattr(self, 'gen_lr', self.optim_lst['warmup'].param_groups[0]['lr'])  # for recorder
            if flag_step:
                self.optim_lst['warmup'].step()
                self.optim_lst['warmup'].zero_grad()
        else:
            setattr(self, 'gen_lr', self.optim_lst['gen'].param_groups[0]['lr'])  # for recorder
            if flag_step:
                self.optim_lst['gen'].step()
                self.optim_lst['gen'].zero_grad()

    def update_params(
            self,
            data,
            iter,
            flag_step,
            inter_step,
            additional=None
        ):
        self.gen_loss = 0.  # for recorder
        self.dis_loss = 0.
        self.gen_lr = None
        self.dis_lr = None
        self.gen_im_lst = dict()

        self.CharbonnierLoss = None
        self.niter_warmup = self.opts_dict['train']['niter_warmup']

        if iter < self.niter_warmup:
            self.update_gen_params(data=data, flag_step=flag_step, inter_step=inter_step, if_warmup=True)
            
            msg = (
                f'gen_lr: [{self.gen_lr:.3e}]; '
                f'gen_loss: [{self.gen_loss:.3e}]; '
                )
            write_dict_lst = [
                dict(tag='gen_loss', scalar=self.gen_loss),
                dict(tag='gen_lr', scalar=self.gen_lr),
                ]
            write_dict_lst.append(dict(
                tag='CharbonnierLoss',
                scalar=self.CharbonnierLoss,
                ))
            msg += f'CharbonnierLoss: [{self.CharbonnierLoss:.3e}]; '
        else:
            for param in self.model.module_lst['gen'].parameters():
                param.requires_grad = True
            for param in self.model.module_lst['dis'].parameters():
                param.requires_grad = False  # turn off grads of the discriminator
            self.update_gen_params(data=data, flag_step=flag_step, inter_step=inter_step)
            
            for param in self.model.module_lst['gen'].parameters():
                param.requires_grad = False
            for param in self.model.module_lst['dis'].parameters():
                param.requires_grad = True
            self.update_dis_params(data=data, flag_step=flag_step, inter_step=inter_step)

            msg = (
                f'dis_lr: [{self.dis_lr:.3e}]; gen_lr: [{self.gen_lr:.3e}]; '
                f'dis_real_loss: [{self.dis_real_loss:.3e}]; dis_fake_loss: [{self.dis_fake_loss:.3e}]; gen_loss: [{self.gen_loss:.3e}]; '
                )
            write_dict_lst = [
                dict(tag='dis_real_loss', scalar=self.dis_real_loss), dict(tag='dis_fake_loss', scalar=self.dis_fake_loss),
                dict(tag='gen_loss', scalar=self.gen_loss),
                dict(tag='dis_lr', scalar=self.dis_lr), dict(tag='gen_lr', scalar=self.gen_lr),
                ]
            for loss_item in self.loss_lst:
                write_dict_lst.append(dict(
                    tag=loss_item,
                    scalar=getattr(self, loss_item),
                    ))
                msg += f'{loss_item}: [{getattr(self, loss_item):.3e}]; '
        return msg[:-2], write_dict_lst, self.gen_im_lst
