# Inherited from mmediting/configs/restorers/basicvsr_plusplus/
# basicvsr_plusplus_c64n7_8x1_600k_reds4
bs = 8
ngpus = 2
assert bs % ngpus == 0, ('Samples in a batch should better be evenly'
                         ' distributed among all GPUs.')

nf = 64
nb = 7
ps = 256
niter_k = 600

exp_name = (f'basicvsr_plus_plus_vimeo90k_triplet'
            f'_nf{nf}_nb{nb}'
            f'_ps{ps}_bs{bs}_{niter_k}k_g{ngpus}')

rescale = 1  # must be 2^n

# model settings
model = dict(
    type='BasicRestorerVQESequence',
    generator=dict(
        type='BasicVSRPlusPlus',
        mid_channels=nf,
        num_blocks=nb,
        is_low_res_input=False,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))

# model training and testing settings
train_cfg = dict(fix_iter=5000, fix_module=['edvr', 'spynet'])
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=rescale)

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='lq',
         flag='unchanged'),
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='gt',
         flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=ps),
    dict(type='Flip',
         keys=['lq', 'gt'],
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
]
test_pipeline = [
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='lq',
         flag='unchanged'),
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='gt',
         flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect',
         keys=['lq', 'gt'],
         meta_keys=['lq_path', 'gt_path', 'key']),
]

dataset_type = 'Vimeo90KTripletSequenceDataset'
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
optimizers = dict(generator=dict(
    type='Adam',
    lr=1e-4,
    betas=(0.9, 0.99),
    paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)}),
))

# learning policy
total_iters = niter_k * 1000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[total_iters],
    restart_weights=[1],
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
find_unused_parameters = True  # for spynet pre-training
