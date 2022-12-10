exp_name = 'esrgan_stage1'
scale = 1

# model settings
model = dict(type='BasicRestorerQE',
             generator=dict(type='RRDBNetQE',
                            in_channels=3,
                            out_channels=3,
                            mid_channels=64,
                            num_blocks=23,
                            growth_channels=32,
                            upscale_factor=scale),
             pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))

# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile',
         io_backend='disk',
         key='lq',
         flag='unchanged'),
    dict(type='LoadImageFromFile',
         io_backend='disk',
         key='gt',
         flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize',
         keys=['lq', 'gt'],
         mean=[0, 0, 0],
         std=[1, 1, 1],
         to_rgb=True),
    dict(type='PairedRandomCrop', gt_patch_size=128),
    dict(type='Flip',
         keys=['lq', 'gt'],
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
test_pipeline = [
    dict(type='LoadImageFromFile',
         io_backend='disk',
         key='lq',
         flag='unchanged'),
    dict(type='LoadImageFromFile',
         io_backend='disk',
         key='gt',
         flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize',
         keys=['lq', 'gt'],
         mean=[0, 0, 0],
         std=[1, 1, 1],
         to_rgb=True),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(workers_per_gpu=8,
            train_dataloader=dict(samples_per_gpu=8, drop_last=True),
            val_dataloader=dict(samples_per_gpu=1),
            test_dataloader=dict(samples_per_gpu=1),
            train=dict(type='RepeatDataset',
                       times=1000,
                       dataset=dict(type='QEFolderDataset',
                                    lq_folder='./data/div2k/train/lq',
                                    gt_folder='./data/div2k/train/gt',
                                    pipeline=train_pipeline,
                                    filename_tmpl='{}.png')),
            val=dict(type='QEFolderDataset',
                     lq_folder='./data/div2k/valid/lq',
                     gt_folder='./data/div2k/valid/gt',
                     pipeline=test_pipeline,
                     filename_tmpl='{}.png'),
            test=dict(type='QEFolderDataset',
                      lq_folder='./data/div2k/valid/lq',
                      gt_folder='./data/div2k/valid/gt',
                      pipeline=test_pipeline,
                      filename_tmpl='{}.png'))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 500000
lr_config = dict(policy='CosineRestart',
                 by_epoch=False,
                 periods=[total_iters],
                 min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit-sr'))
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
