bs = 16
ngpus = 2
assert bs % ngpus == 0, ('Samples in a batch should better be evenly'
                         ' distributed among all GPUs.')

nf = 64
nb = 23
gf = 32
ps = 128
niter_k = 400

pretrained = ('work_dirs/'
              'esrgan_div2k_stage1_nf64_nb23_gf32_ps128_bs16_500k_g2/'
              'iter_500000.pth')

lr_steps = [50, 100, 200, 300]
lr_steps = [s * 1000 for s in lr_steps]

exp_name = (f'esrgan_div2k_stage2_nf{nf}_nb{nb}_gf{gf}'
            f'_ps{ps}_bs{bs}_{niter_k}k_g{ngpus}')

scale = 1

# model settings
model = dict(
    type='ESRGANQE',
    generator=dict(type='RRDBNetQE',
                   in_channels=3,
                   out_channels=3,
                   mid_channels=nf,
                   num_blocks=nb,
                   growth_channels=gf,
                   upscale_factor=scale),
    discriminator=dict(type='ModifiedVGG', in_channels=3, mid_channels=64),
    pixel_loss=dict(type='L1Loss', loss_weight=1e-2, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'34': 1.0},
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False,
        pretrained='https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'),
    gan_loss=dict(type='GANLoss',
                  gan_type='vanilla',
                  loss_weight=5e-3,
                  real_label_val=1.0,
                  fake_label_val=0),
    pretrained=pretrained,
)

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
    dict(type='PairedRandomCrop', gt_patch_size=ps),
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

data = dict(workers_per_gpu=bs // ngpus,
            train_dataloader=dict(samples_per_gpu=bs // ngpus, drop_last=True),
            val_dataloader=dict(samples_per_gpu=1),
            test_dataloader=dict(samples_per_gpu=1),
            train=dict(type='RepeatDataset',
                       times=1000,
                       dataset=dict(type='PairedSameSizeImageDataset',
                                    lq_folder='data/div2k/train/lq',
                                    gt_folder='data/div2k/train/gt',
                                    pipeline=train_pipeline,
                                    filename_tmpl='{}.png',
                                    test_mode=False)),
            val=dict(type='PairedSameSizeImageDataset',
                     lq_folder='data/div2k/valid/lq',
                     gt_folder='data/div2k/valid/gt',
                     pipeline=test_pipeline,
                     filename_tmpl='{}.png',
                     test_mode=True),
            test=dict(type='PairedSameSizeImageDataset',
                      lq_folder='data/div2k/valid/lq',
                      gt_folder='data/div2k/valid/gt',
                      pipeline=test_pipeline,
                      filename_tmpl='{}.png',
                      test_mode=True))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)),
                  discriminator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = niter_k * 1000
lr_config = dict(policy='Step', by_epoch=False, step=lr_steps, gamma=0.5)

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
work_dir = f'work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
