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

model = dict(type='BasicRestorerQE',
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

patchsize = 48
train_cfg = None
test_cfg = dict(
    metrics=['PSNR', 'SSIM'],
    crop_border=1,
    unfolding=dict(patchsize=patchsize, splits=16)  # to save memory
)

train_pipeline = [
    dict(type='LoadImageFromFileMultiKeys',
         io_backend='disk',
         keys=['lq', 'gt'],
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCropQE', patch_size=patchsize, keys=['lq', 'gt']),
    dict(type='Flip',
         keys=['lq', 'gt'],
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]
valid_pipeline = [
    dict(type='LoadImageFromFileMultiKeys',
         io_backend='disk',
         keys=['lq', 'gt'],
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedCenterCrop', patch_size=patchsize, keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]
test_pipeline = [
    dict(type='LoadImageFromFileMultiKeys',
         io_backend='disk',
         keys=['lq', 'gt'],
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

batchsize = 16
ngpus = 1
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
                     pipeline=valid_pipeline,
                     lq_ext='.png',
                     test_mode=True),
            test=dict(type=dataset_type,
                      lq_folder='data/div2k/valid/lq',
                      gt_folder='data/div2k/valid/gt',
                      pipeline=test_pipeline,
                      lq_ext='.png',
                      test_mode=True))

optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))
total_iters = 500 * 1000
lr_config = dict(policy='CosineRestart',
                 by_epoch=False,
                 periods=[total_iters],
                 min_lr=1e-6)

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
