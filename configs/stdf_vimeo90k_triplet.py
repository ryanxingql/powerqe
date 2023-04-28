exp_name = 'stdf_vimeo90k_triplet'

params = dict(batchsize=32,
              ngpus=4,
              patchsize=128,
              kiters=1000,
              radius=1,
              nchannels=[3, 32, 64, 48],
              nblocks=[3, 6])

model = dict(type='BasicRestorerVQE',
             generator=dict(
                 type='STDFNet',
                 io_channels=params['nchannels'][0],
                 radius=params['radius'],
                 nf_stdf=params['nchannels'][1],
                 nb_stdf=params['nblocks'][0],
                 nf_stdf_out=params['nchannels'][2],
                 nf_qe=params['nchannels'][3],
                 nb_qe=params['nblocks'][1],
             ),
             pixel_loss=dict(type='CharbonnierLoss',
                             loss_weight=1.0,
                             reduction='mean'))

train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=1)

train_pipeline = [
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='lq',
         channel_order='rgb',
         backend='pillow'),
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='gt',
         channel_order='rgb',
         backend='pillow'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCropQE',
         patch_size=params['patchsize'],
         keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect',
         keys=['lq', 'gt'],
         meta_keys=['lq_path', 'gt_path', 'key'])
]
test_pipeline = [
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='lq',
         channel_order='rgb',
         backend='pillow'),
    dict(type='LoadImageFromFileList',
         io_backend='disk',
         key='gt',
         channel_order='rgb',
         backend='pillow'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect',
         keys=['lq', 'gt'],
         meta_keys=['lq_path', 'gt_path', 'key'])
]

assert params['batchsize'] % params['ngpus'] == 0, (
    'Samples in a batch should better be evenly'
    ' distributed among all GPUs.')
dataset_type = 'PairedSameSizeVimeo90KTripletDataset'
dataset_gt_dir = 'data/vimeo_triplet'
dataset_lq_dir = 'data/vimeo_triplet_lq'
batchsize_gpu = params['batchsize'] // params['ngpus']
data = dict(workers_per_gpu=batchsize_gpu,
            train_dataloader=dict(samples_per_gpu=batchsize_gpu,
                                  drop_last=True),
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
                           filename_tmpl='{}.png',
                           edge_padding=True,
                           center_gt=True)),
            val=dict(type=dataset_type,
                     folder=f'{dataset_lq_dir}/sequences',
                     gt_folder=f'{dataset_gt_dir}/sequences',
                     ann_file=f'{dataset_lq_dir}/tri_validlist.txt',
                     pipeline=test_pipeline,
                     test_mode=True,
                     filename_tmpl='{}.png',
                     edge_padding=True,
                     center_gt=True),
            test=dict(type=dataset_type,
                      folder=f'{dataset_lq_dir}/sequences',
                      gt_folder=f'{dataset_gt_dir}/sequences',
                      ann_file=f'{dataset_lq_dir}/tri_testlist.txt',
                      pipeline=test_pipeline,
                      test_mode=True,
                      filename_tmpl='{}.png',
                      edge_padding=True,
                      center_gt=True))

optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

total_iters = params['kiters'] * 1000
lr_config = dict(policy='CosineRestart',
                 by_epoch=False,
                 periods=[total_iters],
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
