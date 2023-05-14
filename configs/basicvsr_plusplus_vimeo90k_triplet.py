"""Inherited from mmediting/configs/restorers/basicvsr_plusplus/
basicvsr_plusplus_c64n7_8x1_600k_reds4."""

exp_name = 'basicvsr_plus_plus_vimeo90k_triplet'

center_gt = False
model = dict(
    type='BasicRestorerVQESequence',
    generator=dict(
        type='BasicVSRPlusPlus',
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=False,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    center_gt=center_gt)

train_cfg = dict(fix_iter=5000, fix_module=['edvr', 'spynet'])
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=1)

train_pipeline = [
    dict(type='LoadImageFromFileListMultiKeys',
         io_backend='disk',
         keys=['lq', 'gt'],
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCropQE', patch_size=256, keys=['lq', 'gt']),
    dict(type='Flip',
         keys=['lq', 'gt'],
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect',
         keys=['lq', 'gt'],
         mmeta_keys=['lq_path', 'gt_path', 'key'])
]
test_pipeline = [
    dict(type='LoadImageFromFileListMultiKeys',
         io_backend='disk',
         keys=['lq', 'gt'],
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect',
         keys=['lq', 'gt'],
         meta_keys=['lq_path', 'gt_path', 'key'])
]

batchsize = 8
ngpus = 2
assert batchsize % ngpus == 0, ('Samples in a batch should better be evenly'
                                ' distributed among all GPUs.')
dataset_type = 'PairedSameSizeVideoDataset'
dataset_gt_root = 'data/vimeo_triplet'
dataset_lq_folder = 'data/vimeo_triplet_lq'
batchsize_gpu = batchsize // ngpus
data = dict(workers_per_gpu=batchsize_gpu,
            train_dataloader=dict(samples_per_gpu=batchsize_gpu,
                                  drop_last=True),
            val_dataloader=dict(samples_per_gpu=1),
            test_dataloader=dict(samples_per_gpu=1),
            train=dict(type='RepeatDataset',
                       times=1000,
                       dataset=dict(
                           type=dataset_type,
                           lq_folder=f'{dataset_lq_folder}',
                           gt_folder=f'{dataset_gt_root}/sequences',
                           ann_file=f'{dataset_gt_root}/tri_trainlist.txt',
                           pipeline=train_pipeline,
                           test_mode=False,
                           lq_ext='.png',
                           samp_len=-1,
                           edge_padding=False,
                           center_gt=center_gt)),
            val=dict(type=dataset_type,
                     folder=f'{dataset_lq_folder}',
                     gt_folder=f'{dataset_gt_root}/sequences',
                     ann_file=f'{dataset_gt_root}/tri_validlist.txt',
                     pipeline=test_pipeline,
                     test_mode=True,
                     lq_ext='.png',
                     samp_len=-1,
                     edge_padding=False,
                     center_gt=center_gt),
            test=dict(type=dataset_type,
                      folder=f'{dataset_lq_folder}',
                      gt_folder=f'{dataset_gt_root}/sequences',
                      ann_file=f'{dataset_gt_root}/tri_testlist.txt',
                      pipeline=test_pipeline,
                      test_mode=True,
                      lq_ext='.png',
                      samp_len=-1,
                      edge_padding=False,
                      center_gt=center_gt))

optimizers = dict(
    generator=dict(type='Adam',
                   lr=1e-4,
                   betas=(0.9, 0.99),
                   paramwise_cfg=dict(
                       custom_keys={'spynet': dict(lr_mult=0.25)})))

total_iters = 500 * 1000
lr_config = dict(policy='CosineRestart',
                 by_epoch=False,
                 periods=[total_iters],
                 restart_weights=[1],
                 min_lr=1e-7)

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
find_unused_parameters = True  # for spynet pre-training
