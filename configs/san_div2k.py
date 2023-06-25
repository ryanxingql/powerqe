_base_ = ['_base_/runtime.py', '_base_/div2k.py']

exp_name = 'san_div2k'

patch_size = 48

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

test_cfg = dict(unfolding=dict(patchsize=patch_size,
                               splits=16))  # to save memory

train_pipeline = [
    dict(type='LoadImageFromFile',
         io_backend='disk',
         key='lq',
         channel_order='rgb'),
    dict(type='LoadImageFromFile',
         io_backend='disk',
         key='gt',
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop',
         gt_patch_size=patch_size),  # keys must be 'lq' and 'gt'
    dict(type='Flip',
         keys=['lq', 'gt'],
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))

work_dir = f'work_dirs/{exp_name}'
