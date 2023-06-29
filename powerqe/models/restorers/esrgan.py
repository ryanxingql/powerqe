"""Copyright (c) OpenMMLab. All rights reserved.

Copyright 2023 RyanXingQL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at
https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from mmedit.models.common import set_requires_grad

from ..builder import build_component, build_loss
from ..registry import MODELS
from .basic_restorer import BasicQERestorer


@MODELS.register_module()
class ESRGANRestorer(BasicQERestorer):
    """ESRGAN restorer for quality enhancement.

    Args:
        generator (dict): Config for the generator.
        discriminator (dict): Config for the discriminator. Default: None.
        gan_loss (dict): Config for the GAN loss.
            Note that the loss weight in GAN loss is only for the generator.
        pixel_loss (dict): Config for the pixel loss. Default: None.
        perceptual_loss (dict): Config for the perceptual loss. Default: None.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of GAN by setting:
                disc_steps: how many discriminator updates after one generate
                    update;
                disc_init_steps: how many discriminator updates at the start of
                    the training.

    These two keys are useful when training with WGAN.
        test_cfg (dict): Config for testing.
            Default: None.
        pretrained (str): Path for pretrained model.
            Default: None.
    """

    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_loss=None,
                 pixel_loss=None,
                 perceptual_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator=generator,
                         pixel_loss=pixel_loss,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         pretrained=pretrained)

        # generator is defined in the __init__() of BasicRestorer

        # discriminator
        self.discriminator = build_component(
            discriminator) if discriminator else None

        # loss
        # pixel_loss is defined in the __init__() of BasicRestorer
        self.gan_loss = build_loss(gan_loss) if gan_loss else None
        self.perceptual_loss = build_loss(
            perceptual_loss) if perceptual_loss else None

        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))
        self.step_counter = 0  # counting training steps

    def init_weights(self, pretrained=None):
        """Init the generator weights using the generator's method.

        Therefore r'^generator.' must be removed.

        Args:
            pretrained (str, optional): Path for pretrained weights.
                If given None, pretrained weights will not be loaded.
        """
        self.generator.init_weights(pretrained=pretrained,
                                    revise_keys=[(r'^generator\.', ''),
                                                 (r'^module\.', '')])
        # if self.discriminator:
        #     self.discriminator.init_weights(pretrained=pretrained)

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
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
