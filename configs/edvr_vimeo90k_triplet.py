# Inherited from mmediting/configs/restorers/edvr/
# edvrm_wotsa_x4_g8_600k_reds.py to avoid pre-training of tsa
# Decrease patch size from 256 to 128 to save memory.
bs = 32
ngpus = 2
assert bs % ngpus == 0, ('Samples in a batch should better be evenly'
                         ' distributed among all GPUs.')

radius = 1
nf = 64
nbs = [5, 10]
ps = 128
niter_k = 600
lr_periods = [150000, 150000, 150000, 150000]

exp_name = (f'edvr_vimeo90k_triplet'
            f'_nf{nf}_r{radius}_nb{nbs[0]}_{nbs[1]}'
            f'_ps{ps}_bs{bs}_{niter_k}k_g{ngpus}')

rescale = 1  # must be 2^n

# model settings
model = dict(
    type='BasicRestorerVQE',
    generator=dict(
        type='EDVRNetQE',
        in_channels=3,
        out_channels=3,
        mid_channels=nf,
        num_frames=2 * radius + 1,
        deform_groups=8,
        num_blocks_extraction=nbs[0],
        num_blocks_reconstruction=nbs[1],
        center_frame_idx=1,  # invalid when TSA is off
        with_tsa=False,
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='sum'))

# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=rescale)

# dataset settings
train_pipeline = [
    # dict(
    #     type='GenerateFrameIndices',
    #     interval_list=[1],
    #     frames_per_clip=99,
    # ),  # no need for Vimeo90KTripletCenterGTDataset
    # dict(
    #     type='TemporalReverse',
    #     keys='lq_path',
    #     reverse_ratio=0,
    # ),  # no need for compressed video
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='lq',
         flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged',
    ),  # gt is a single image for Vimeo90KTripletCenterGTDataset
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize',
         keys=['lq', 'gt'],
         mean=[0, 0, 0],
         std=[1, 1, 1],
         to_rgb=True),
    dict(type='PairedRandomCrop', gt_patch_size=ps),
    dict(type='Flip',
         keys=['lq', 'gt'],
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq']),
    dict(
        type='ImageToTensor',
        keys=['gt'],
    ),  # gt is a single image for Vimeo90KTripletCenterGTDataset
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
]
test_pipeline = [
    # dict(
    #     type='GenerateFrameIndiceswithPadding',
    #     padding='reflection_circle',
    # ),  # no need for Vimeo90KTripletCenterGTDataset
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='lq',
         flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged',
    ),  # gt is a single image for Vimeo90KTripletCenterGTDataset
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize',
         keys=['lq', 'gt'],
         mean=[0, 0, 0],
         std=[1, 1, 1],
         to_rgb=True),
    dict(type='FramesToTensor', keys=['lq']),
    dict(
        type='ImageToTensor',
        keys=['gt'],
    ),  # gt is a single image for Vimeo90KTripletCenterGTDataset
    dict(type='Collect',
         keys=['lq', 'gt'],
         meta_keys=['lq_path', 'gt_path', 'key']),
]

dataset_type = 'Vimeo90KTripletCenterGTDataset'
dataset_gt_dir = 'data/vimeo_triplet'
dataset_lq_dir = 'data/vimeo_triplet_lq'

data = dict(workers_per_gpu=bs // ngpus,
            train_dataloader=dict(samples_per_gpu=bs // ngpus, drop_last=True),
            val_dataloader=dict(samples_per_gpu=1),
            test_dataloader=dict(samples_per_gpu=1),
            train=dict(type='RepeatDataset',
                       times=1000,
                       dataset=dict(
                           type=dataset_type,
                           folder=f'{dataset_lq_dir}',
                           gt_folder=f'{dataset_gt_dir}/sequences',
                           ann_file=f'{dataset_gt_dir}/tri_trainlist.txt',
                           pipeline=train_pipeline,
                           test_mode=False,
                           filename_tmpl='{}.png')),
            val=dict(type=dataset_type,
                     folder=f'{dataset_lq_dir}',
                     gt_folder=f'{dataset_gt_dir}/sequences',
                     ann_file=f'{dataset_gt_dir}/tri_validlist.txt',
                     pipeline=test_pipeline,
                     test_mode=True,
                     filename_tmpl='{}.png'),
            test=dict(type=dataset_type,
                      folder=f'{dataset_lq_dir}',
                      gt_folder=f'{dataset_gt_dir}/sequences',
                      ann_file=f'{dataset_gt_dir}/tri_testlist.txt',
                      pipeline=test_pipeline,
                      test_mode=True,
                      filename_tmpl='{}.png'))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=4e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = niter_k * 1000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=lr_periods,
    restart_weights=[1, 0.5, 0.5, 0.5],
    min_lr=1e-7,
)

checkpoint_config = dict(
    interval=5000,
    save_optimizer=True,
    by_epoch=False,
)
evaluation = dict(
    interval=5000,
    save_image=False,
    gpu_collect=True,
)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ],
)
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
