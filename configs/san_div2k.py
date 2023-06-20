_base_ = ['_base_/runtime.py', '_base_/div2k.py']

exp_name = 'san_div2k'

params = dict(batchsize=16,
              ngpus=1,
              patchsize=48,
              kiters=300,
              nchannels=[3, 64],
              reduction=16,
              nblocks=10,
              ngroups=20,
              kernelsize=3,
              resscale=1)

model = dict(type='BasicQERestorer',
             generator=dict(type='SAN',
                            n_resgroups=20,
                            n_resblocks=10,
                            n_feats=64,
                            kernel_size=3,
                            reduction=16,
                            scale=1,
                            rgb_range=1,
                            n_colors=3,
                            res_scale=1),
             pixel_loss=dict(type='CharbonnierLoss',
                             loss_weight=1.0,
                             reduction='mean'))

test_cfg = dict(unfolding=dict(patchsize=48, splits=16))  # to save memory

work_dir = f'work_dirs/{exp_name}'
