bs = 32
ngpus = 4
assert bs % ngpus == 0

radius = 1
nfs = [32, 64, 48]
nbs = [3, 6]
ps = 128
niter_k = 1000

exp_name = (f'stdf_vimeo90k_triplet'
            f'_r{radius}_nf{nfs[0]}_{nfs[1]}_{nfs[2]}_nb{nbs[0]}_{nbs[1]}'
            f'_ps{ps}_bs{bs}_{niter_k}k_g{ngpus}')

rescale = 1  # must be 2^n

# model settings
model = dict(type='BasicRestorerVQE',
             generator=dict(
                 type='STDFNet',
                 radius=radius,
                 nf_stdf=nfs[0],
                 nb_stdf=nbs[0],
                 nf_stdf_out=nfs[1],
                 nf_qe=nfs[2],
                 nb_qe=nbs[1],
             ),
             pixel_loss=dict(type='CharbonnierLoss',
                             loss_weight=1.0,
                             reduction='mean'))

# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=rescale)

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='lq',
         channel_order='rgb',
         backend='pillow'),
    dict(type='LoadImageFromFile',
         io_backend='disk',
         key='gt',
         channel_order='rgb',
         backend='pillow'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='ImageToTensor', keys=['gt']),
    dict(type='Collect',
         keys=['lq', 'gt'],
         meta_keys=['lq_path', 'gt_path', 'key'])
]
test_pipeline = train_pipeline

dataset_type = 'Vimeo90KTripletCenterGTDataset'
dataset_gt_dir = './data/vimeo_triplet'
dataset_lq_dir = './data/vimeo_triplet_x265'

data = dict(workers_per_gpu=bs // ngpus,
            train_dataloader=dict(samples_per_gpu=bs // ngpus, drop_last=True),
            val_dataloader=dict(samples_per_gpu=1),
            test_dataloader=dict(samples_per_gpu=1),
            train=dict(type='RepeatDataset',
                       times=1000,
                       dataset=dict(
                           type=dataset_type,
                           folder=f'{dataset_lq_dir}/sequences',
                           gt_folder=f'{dataset_gt_dir}/sequences',
                           ann_file=f'{dataset_lq_dir}/tri_trainlist.txt',
                           pipeline=train_pipeline,
                           test_mode=False,
                           filename_tmpl='{}.png')),
            val=dict(type=dataset_type,
                     folder=f'{dataset_lq_dir}/sequences',
                     gt_folder=f'{dataset_gt_dir}/sequences',
                     ann_file=f'{dataset_lq_dir}/tri_validlist.txt',
                     pipeline=test_pipeline,
                     test_mode=True,
                     filename_tmpl='{}.png'),
            test=dict(type=dataset_type,
                      folder=f'{dataset_lq_dir}/sequences',
                      gt_folder=f'{dataset_gt_dir}/sequences',
                      ann_file=f'{dataset_lq_dir}/tri_testlist.txt',
                      pipeline=test_pipeline,
                      test_mode=True,
                      filename_tmpl='{}.png'))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = niter_k * 1000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[total_iters],
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
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
