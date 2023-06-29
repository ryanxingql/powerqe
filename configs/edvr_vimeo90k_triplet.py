# mmediting/configs/restorers/edvr/
# edvrm_wotsa_x4_g8_600k_reds.py
_base_ = ['_base_/runtime.py', '_base_/vimeo90k_triplet.py']

exp_name = 'edvr_vimeo90k_triplet'

center_gt = True
model = dict(
    type='BasicVQERestorer',
    generator=dict(
        type='EDVRNetQE',
        io_channels=3,
        mid_channels=64,
        num_frames=3,
        deform_groups=8,
        num_blocks_extraction=5,
        num_blocks_reconstruction=10,
        center_frame_idx=1,  # invalid when TSA is off
        with_tsa=False),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    center_gt=center_gt)

norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1])
test_cfg = dict(denormalize=norm_cfg)

train_pipeline = [
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='lq',
         channel_order='rgb'),
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='gt',
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], **norm_cfg),
    dict(type='PairedRandomCrop',
         gt_patch_size=256),  # keys must be 'lq' and 'gt'
    dict(type='Flip',
         keys=['lq', 'gt'],
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]
test_pipeline = [
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='lq',
         channel_order='rgb'),
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='gt',
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], **norm_cfg),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect',
         keys=['lq', 'gt'],
         meta_keys=['lq_path', 'gt_path', 'key'])
]

data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline, center_gt=center_gt)),
    val=dict(pipeline=test_pipeline, center_gt=center_gt),
    test=dict(pipeline=test_pipeline, center_gt=center_gt))

work_dir = f'work_dirs/{exp_name}'
