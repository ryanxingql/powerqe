exp_name = 'mfqev2_vimeo90k_triplet'

model = dict(
    type='BasicRestorerVQE',
    generator=dict(
        type='MFQEv2',
        io_channels=3,
        nf=32,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))

train_cfg = dict(fix_iter=5000, fix_module=['spynet'])
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=1)

train_pipeline = [
    dict(type='LoadImageFromFileListMultiKeys',
         io_backend='disk',
         keys=['lq', 'gt'],
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCropQE', patch_size=256, keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect',
         keys=['lq', 'gt'],
         meta_keys=['lq_path', 'gt_path', 'key'])
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

batchsize = 32
ngpus = 2
assert batchsize % ngpus == 0, ('Samples in a batch should better be evenly'
                                ' distributed among all GPUs.')
dataset_type = 'PairedSameSizeVimeo90KTripletDatasetWithQP'
dataset_gt_dir = 'data/vimeo_triplet'
dataset_lq_dir = 'data/vimeo_triplet_lq'
qp_info = dict(qp=37,
               intra_qp_offset=-1,
               qp_offset=[5, 4] * 3 + [5, 1],
               qp_offset_model_off=[-6.5] * 7 + [0],
               qp_offset_model_scale=[0.2590] * 7 + [0])
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
                           qp_info=qp_info,
                           folder=f'{dataset_lq_dir}',
                           gt_folder=f'{dataset_gt_dir}/sequences',
                           ann_file=f'{dataset_gt_dir}/tri_trainlist.txt',
                           pipeline=train_pipeline,
                           test_mode=False,
                           lq_ext='.png',
                           edge_padding=True,
                           center_gt=True)),
            val=dict(type=dataset_type,
                     qp_info=qp_info,
                     folder=f'{dataset_lq_dir}',
                     gt_folder=f'{dataset_gt_dir}/sequences',
                     ann_file=f'{dataset_gt_dir}/tri_validlist.txt',
                     pipeline=test_pipeline,
                     test_mode=True,
                     lq_ext='.png',
                     edge_padding=True,
                     center_gt=True),
            test=dict(type=dataset_type,
                      qp_info=qp_info,
                      folder=f'{dataset_lq_dir}',
                      gt_folder=f'{dataset_gt_dir}/sequences',
                      ann_file=f'{dataset_gt_dir}/tri_testlist.txt',
                      pipeline=test_pipeline,
                      test_mode=True,
                      lq_ext='.png',
                      edge_padding=True,
                      center_gt=True))

optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

total_iters = 500 * 1000
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
find_unused_parameters = True  # for spynet pre-training
