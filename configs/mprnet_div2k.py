bs = 16
ngpus = 1
assert bs % ngpus == 0

nf = 96
ps = 128
niter_k = 300

exp_name = f'mprnet_div2k_nf{nf}_ps{ps}_bs{bs}_{niter_k}k_g{ngpus}'

# model settings
model = dict(type='BasicRestorerQE',
             generator=dict(
                 type='MPRNet',
                 in_c=3,
                 out_c=3,
                 n_feat=nf,
             ),
             pixel_loss=dict(
                 type='CharbonnierLoss',
                 loss_weight=1.0,
                 reduction='mean',
             ))

# model training and testing settings
train_cfg = None
test_cfg = dict(
    metrics=['PSNR', 'SSIM'],
    crop_border=1,
    unfolding=dict(
        patch_sz=ps,
        splits=4,
    )  # to save memory for testing
)

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
valid_pipeline = [
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
    # dict(type='PairedCenterCrop', gt_patch_size=ps),
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
                     pipeline=valid_pipeline,
                     filename_tmpl='{}.png',
                     test_mode=True),
            test=dict(type='PairedSameSizeImageDataset',
                      lq_folder='data/div2k/valid/lq',
                      gt_folder='data/div2k/valid/gt',
                      pipeline=test_pipeline,
                      filename_tmpl='{}.png',
                      test_mode=True))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = niter_k * 1000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[total_iters],
    min_lr=1e-6,
)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
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
work_dir = f'work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
