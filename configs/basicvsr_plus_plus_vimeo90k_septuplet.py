"""
Ref: mmediting/configs/restorers/basicvsr_plusplus
/basicvsr_plusplus_c64n7_8x1_600k_reds4.py
"""
_base_ = ['_base_/runtime.py', '_base_/vimeo90k_septuplet.py']

exp_name = 'basicvsr_plus_plus_vimeo90k_septuplet'

center_gt = False
model = dict(
    type='BasicVQERestorer',
    generator=dict(
        type='BasicVSRPlusPlus',
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=False,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    center_gt=center_gt)

train_cfg = dict(_delete_=True, fix_iter=5000, fix_module=['edvr', 'spynet'])

data = dict(train=dict(dataset=dict(edge_padding=False, center_gt=center_gt)),
            val=dict(edge_padding=False, center_gt=center_gt),
            test=dict(edge_padding=False, center_gt=center_gt))

optimizers = dict(generator=dict(paramwise_cfg=dict(
    custom_keys={'spynet': dict(lr_mult=0.25)})))

work_dir = f'work_dirs/{exp_name}'
find_unused_parameters = True  # for spynet pre-training
