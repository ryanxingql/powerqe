import net

from utils import BaseAlg, return_optimizer


class ESRGANAlgorithm(BaseAlg):
    def __init__(self, opts_dict, if_train, if_dist):
        model_cls = getattr(net, 'ESRGANModel')  # FIXME
        super().__init__(opts_dict=opts_dict, model_cls=model_cls, if_train=if_train, if_dist=if_dist)

    def accum_gradient(self, module, stage, group, data, inter_step, **_):
        data_lq = data['lq'].cuda(non_blocking=True)
        data_gt = data['gt'].cuda(non_blocking=True)
        data_out = module(inp_t=data_lq)

        num_show = 3
        if group == 'gen':
            self._im_lst = dict(
                data_lq=data['lq'][:num_show],
                data_gt=data['gt'][:num_show],
                generated=data_out.detach()[:num_show].cpu().clamp_(0., 1.),
            )  # for torch.utils.tensorboard.writer.SummaryWriter.add_images: (N C H W) tensor is ok

        loss_total_item = 0.
        for loss_name in self.loss_lst[stage][group]:
            loss_dict = self.loss_lst[stage][group][loss_name]

            if loss_name == 'RelativisticGANLoss':
                loss_tot_, loss_real_, loss_fake_ = loss_dict['fn'](inp=data_out, ref=data_gt, mode=group,
                                                                    weight=(loss_dict['weight'] / float(inter_step)),
                                                                    dis=self.model.net['dis'])
                setattr(self, f'{loss_name}_{group}', loss_tot_)
                setattr(self, f'{loss_name}_real_{group}', loss_real_)
                setattr(self, f'{loss_name}_fake_{group}', loss_fake_)
                loss_total_item += loss_tot_
            else:
                loss_unweighted = loss_dict['fn'](inp=data_out, ref=data_gt)
                setattr(self, f'{loss_name}_{group}', loss_unweighted.item())  # for recorder

                loss_ = loss_dict['weight'] * loss_unweighted / float(inter_step)
                loss_.backward(retain_graph=True)

                loss_total_item += loss_.item()

        setattr(self, f'loss_{group}', loss_total_item)  # for recorder

    def create_optimizer(self, opts_dict):
        for stage in self.training_stage_lst:
            if stage not in self.optim_lst:
                self.optim_lst[stage] = dict()

            for group in opts_dict[stage]:
                if group not in self.optim_lst[stage]:
                    self.optim_lst[stage][group] = dict()

                params = self.model.net['gen'].parameters() if group == 'gen' else self.model.net['dis'].parameters()
                opts_dict_ = dict(
                    name=opts_dict[stage][group]['name'],
                    params=params,
                    opts=opts_dict[stage][group]['opts'],
                )
                optim_ = return_optimizer(**opts_dict_)
                self.optim_lst[stage][group] = optim_  # one optimizer for one group in one stage

    @staticmethod
    def _turn_off_grads(subnet):
        for param in subnet.parameters():
            param.requires_grad = False

    @staticmethod
    def _turn_on_grads(subnet):
        for param in subnet.parameters():
            param.requires_grad = True

    def update_params(self, stage, data, if_step, inter_step, additional):
        msg = ''
        write_dict_lst = []
        for group in self.optim_lst[stage]:
            # Turn off grads of params in other subnets

            subnet = group
            for _subnet in self.model.net:
                if _subnet != subnet:
                    self._turn_off_grads(self.model.net[_subnet])
                else:
                    self._turn_on_grads(self.model.net[_subnet])

            # Accumulate gradients

            self.accum_gradient(module=self.model.net['gen'], stage=stage, group=group, data=data,
                                inter_step=inter_step, additional=additional)

            item_ = getattr(self, f'loss_{group}')
            msg += f'{group} loss: [{item_:.3e}] | '
            write_dict_lst.append(dict(tag=f'loss_{group}', scalar=item_))

            for loss_name in self.loss_lst[stage][group]:
                item_ = getattr(self, f'{loss_name}_{group}')
                msg += f"{loss_name}_{group}: [{item_:.3e}] | "
                write_dict_lst.append(dict(tag=f'{loss_name}_{group}', scalar=item_))

                if loss_name == 'RelativisticGANLoss':
                    for tab in ['real', 'fake']:
                        item_ = getattr(self, f'{loss_name}_{tab}_{group}')
                        msg += f"{loss_name}_{tab}_{group}: [{item_:.3e}] | "
                        write_dict_lst.append(dict(tag=f'{loss_name}_{tab}_{group}', scalar=item_))

            # Update params

            if if_step:
                self.optim_lst[stage][group].step()

                item_ = self.optim_lst[stage][group].param_groups[0]['lr']  # for recorder
                msg += f"lr_{group}: [{item_:.3e}] | "
                write_dict_lst.append(dict(tag=f'lr_{group}', scalar=item_))

                self.optim_lst[stage][group].zero_grad()  # empty the gradients for this group

            # Update learning rate (scheduler)

            if self.if_sched:
                self.sched_lst[stage][group].step()

        return msg[:-3], write_dict_lst, self._im_lst
