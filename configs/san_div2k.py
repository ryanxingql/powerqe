from .script import generate_exp_name

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
exp_name = generate_exp_name('san_div2k', params)
assert params['batchsize'] % params['ngpus'] == 0, (
    'Samples in a batch should better be evenly'
    ' distributed among all GPUs.')

model = dict(type='BasicRestorerQE',
             generator=dict(type='SAN',
                            n_resgroups=params['ngroups'],
                            n_resblocks=params['nblocks'],
                            n_feats=params['nchannels'][1],
                            kernel_size=params['kernelsize'],
                            reduction=params['reduction'],
                            scale=1,
                            rgb_range=1,
                            n_colors=params['nchannels'][0],
                            res_scale=params['resscale']),
             pixel_loss=dict(type='CharbonnierLoss',
                             loss_weight=1.0,
                             reduction='mean'))

train_cfg = None
test_cfg = dict(
    metrics=['PSNR', 'SSIM'],
    crop_border=1,
    unfolding=dict(patchsize=params['patchsize'],
                   splits=16)  # to save memory for testing
)

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
    dict(type='PairedRandomCrop', gt_patch_size=params['patchsize']),
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
    dict(type='PairedCenterCrop', gt_patch_size=params['patchsize']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
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
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

batchsize_gpu = params['batchsize'] // params['ngpus']
data = dict(workers_per_gpu=batchsize_gpu,
            train_dataloader=dict(samples_per_gpu=batchsize_gpu,
                                  drop_last=True),
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

optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))
total_iters = params['kiters'] * 1000
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
