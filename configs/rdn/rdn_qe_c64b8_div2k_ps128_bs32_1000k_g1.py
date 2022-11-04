exp_name = 'rdn_qe_r1c64b8_div2k_ps128_bs32_1000k_g1'

# scale = 1
rescale = 1  # must be 2^n
# model settings
model = dict(
    type='BasicRestorerQE',
    generator=dict(
        type='RDNQE',
        rescale=rescale,
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        # num_blocks=16,
        num_blocks=8),
    # upscale_factor=scale),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))

# model training and testing settings
train_cfg = None
# test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=rescale)

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile',
         io_backend='disk',
         key='lq',
         flag='color',
         channel_order='rgb'),
    dict(type='LoadImageFromFile',
         io_backend='disk',
         key='gt',
         flag='color',
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
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
         flag='color',
         channel_order='rgb'),
    dict(type='LoadImageFromFile',
         io_backend='disk',
         key='gt',
         flag='color',
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    # workers_per_gpu=1,
    workers_per_gpu=32,  # really helpful
    train_dataloader=dict(samples_per_gpu=32, drop_last=True),
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
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 1000000
lr_config = dict(policy='Step',
                 by_epoch=False,
                 step=[200000, 400000, 600000, 800000],
                 gamma=0.5)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# evaluation = dict(interval=5000, save_image=True, gpu_collect=True)
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(interval=100,
                  hooks=[
                      dict(type='TextLoggerHook', by_epoch=False),
                      dict(type='TensorboardLoggerHook'),
                  ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
