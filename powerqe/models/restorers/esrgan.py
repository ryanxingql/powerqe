# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL @2022
import torch
from mmedit.models.common import set_requires_grad

from ..builder import build_backbone, build_component, build_loss
from ..registry import MODELS
from .basic_restorer import BasicRestorerQE


@MODELS.register_module()
class ESRGANQE(BasicRestorerQE):

    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_loss=None,
                 pixel_loss=None,
                 perceptual_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        """
        Similar to the __init__ of SRGAN in mmedit.
        """
        super().__init__(generator=generator,
                         pixel_loss=pixel_loss,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         pretrained=pretrained)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generator
        self.generator = build_backbone(generator)
        self.init_weights(pretrained)

        # discriminator
        self.discriminator = build_component(
            discriminator) if discriminator else None

        # support fp16
        self.fp16_enabled = False

        # loss
        self.gan_loss = build_loss(gan_loss) if gan_loss else None
        self.pixel_loss = build_loss(pixel_loss) if pixel_loss else None
        self.perceptual_loss = build_loss(
            perceptual_loss) if perceptual_loss else None

        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))
        self.step_counter = 0  # counting training steps

    def init_weights(self, pretrained=None):
        """
        Init the generator weights using the generator's method.
            Therefore ^generator. must be removed.
        """
        self.generator.init_weights(pretrained=pretrained,
                                    revise_keys=[(r'^generator\.', ''),
                                                 (r'^module\.', '')])
        # if self.discriminator:
        #     self.discriminator.init_weights(pretrained=pretrained)

    def train_step(self, data_batch, optimizer):
        """
        train_step of ESRGAN in mmedit.
        """
        # data
        lq = data_batch['lq']
        gt = data_batch['gt']

        # generator
        fake_g_output = self.generator(lq)

        losses = dict()
        log_vars = dict()

        # no updates to discriminator parameters.
        set_requires_grad(self.discriminator, False)

        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            if self.pixel_loss:
                losses['loss_pix'] = self.pixel_loss(fake_g_output, gt)
            if self.perceptual_loss:
                loss_percep, loss_style = self.perceptual_loss(
                    fake_g_output, gt)
                if loss_percep is not None:
                    losses['loss_perceptual'] = loss_percep
                if loss_style is not None:
                    losses['loss_style'] = loss_style

            # gan loss for generator
            real_d_pred = self.discriminator(gt).detach()
            fake_g_pred = self.discriminator(fake_g_output)
            loss_gan_fake = self.gan_loss(fake_g_pred -
                                          torch.mean(real_d_pred),
                                          target_is_real=True,
                                          is_disc=False)
            loss_gan_real = self.gan_loss(real_d_pred -
                                          torch.mean(fake_g_pred),
                                          target_is_real=False,
                                          is_disc=False)
            losses['loss_gan'] = (loss_gan_fake + loss_gan_real) / 2

            # parse loss
            loss_g, log_vars_g = self.parse_losses(losses)
            log_vars.update(log_vars_g)

            # optimize
            optimizer['generator'].zero_grad()
            loss_g.backward()
            optimizer['generator'].step()

        # discriminator
        set_requires_grad(self.discriminator, True)
        # real
        fake_d_pred = self.discriminator(fake_g_output).detach()
        real_d_pred = self.discriminator(gt)
        loss_d_real = self.gan_loss(
            real_d_pred - torch.mean(fake_d_pred),
            target_is_real=True,
            is_disc=True
        ) * 0.5  # 0.5 for averaging loss_d_real and loss_d_fake
        loss_d, log_vars_d = self.parse_losses(dict(loss_d_real=loss_d_real))
        optimizer['discriminator'].zero_grad()
        loss_d.backward()
        log_vars.update(log_vars_d)
        # fake
        fake_d_pred = self.discriminator(fake_g_output.detach())
        loss_d_fake = self.gan_loss(
            fake_d_pred - torch.mean(real_d_pred.detach()),
            target_is_real=False,
            is_disc=True
        ) * 0.5  # 0.5 for averaging loss_d_real and loss_d_fake
        loss_d, log_vars_d = self.parse_losses(dict(loss_d_fake=loss_d_fake))
        loss_d.backward()
        log_vars.update(log_vars_d)

        optimizer['discriminator'].step()

        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(log_vars=log_vars,
                       num_samples=len(gt.data),
                       results=dict(lq=lq.cpu(),
                                    gt=gt.cpu(),
                                    output=fake_g_output.cpu()))

        return outputs
