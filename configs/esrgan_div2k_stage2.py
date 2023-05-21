exp_name = 'esrgan_div2k_stage2'

stage1_ckpt = (
    'work_dirs/'
    'esrgan_div2k_stage1_batchsize_16_ngpus_2'
    '_patchsize_128_kiters_500_nchannels_64_nblocks_23_growthfactor_32/'
    'iter_500000.pth')
model = dict(
    type='ESRGANQE',
    generator=dict(type='RRDBNetQE',
                   io_channels=3,
                   mid_channels=64,
                   num_blocks=23,
                   growth_channels=32,
                   upscale_factor=1),
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
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=5e-3,
        real_label_val=1.0,
        fake_label_val=0,
    ),
    pretrained=stage1_ckpt)

train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=1)

train_pipeline = [
    dict(type='LoadImageFromFileMultiKeys',
         io_backend='disk',
         keys=['lq', 'gt'],
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize',
         keys=['lq', 'gt'],
         mean=[0, 0, 0],
         std=[1, 1, 1],
         to_rgb=True),
    dict(type='PairedRandomCropQE', patch_size=128, keys=['lq', 'gt']),
    dict(type='Flip',
         keys=['lq', 'gt'],
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]
test_pipeline = [
    dict(type='LoadImageFromFileMultiKeys',
         io_backend='disk',
         keys=['lq', 'gt'],
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize',
         keys=['lq', 'gt'],
         mean=[0, 0, 0],
         std=[1, 1, 1],
         to_rgb=True),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

batchsize = 16
ngpus = 2
assert batchsize % ngpus == 0, ('Samples in a batch should better be evenly'
                                ' distributed among all GPUs.')
batchsize_gpu = batchsize // ngpus
dataset_type = 'PairedSameSizeImageDataset'
data = dict(workers_per_gpu=batchsize_gpu,
            train_dataloader=dict(samples_per_gpu=batchsize_gpu,
                                  drop_last=True),
            val_dataloader=dict(samples_per_gpu=1),
            test_dataloader=dict(samples_per_gpu=1),
            train=dict(type='RepeatDataset',
                       times=1000,
                       dataset=dict(type=dataset_type,
                                    lq_folder='data/div2k/train/lq',
                                    gt_folder='data/div2k/train/gt',
                                    pipeline=train_pipeline,
                                    lq_ext='.png',
                                    test_mode=False)),
            val=dict(type=dataset_type,
                     lq_folder='data/div2k/valid/lq',
                     gt_folder='data/div2k/valid/gt',
                     pipeline=test_pipeline,
                     lq_ext='.png',
                     test_mode=True),
            test=dict(type=dataset_type,
                      lq_folder='data/div2k/valid/lq',
                      gt_folder='data/div2k/valid/gt',
                      pipeline=test_pipeline,
                      lq_ext='.png',
                      test_mode=True))

optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)),
                  discriminator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

total_iters = 400 * 1000
lr_config = dict(policy='Step',
                 by_epoch=False,
                 step=[s * 1000 for s in [50, 100, 200, 300]],
                 gamma=0.5)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(interval=100,
                  hooks=[
                      dict(type='TextLoggerHook', by_epoch=False),
                      dict(type='TensorboardLoggerHook')
                  ])
visual_config = None

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
