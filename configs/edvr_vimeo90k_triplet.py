"""Inherited from mmediting/configs/restorers/edvr/
edvrm_wotsa_x4_g8_600k_reds.py to avoid the pre-training of TSA.

Decrease patch size from 256 to 128 to save memory.
"""

exp_name = 'edvr_vimeo90k_triplet'

params = dict(batchsize=32,
              ngpus=2,
              patchsize=128,
              kiters=600,
              nchannels=[3, 64],
              nblocks=[5, 10],
              radius=1,
              ngroups=8,
              klrperiods=[150, 150, 150, 150])

model = dict(
    type='BasicRestorerVQE',
    generator=dict(
        type='EDVRNetQE',
        io_channels=params['nchannels'][0],
        mid_channels=params['nchannels'][1],
        num_frames=2 * params['radius'] + 1,
        deform_groups=params['ngroups'],
        num_blocks_extraction=params['nblocks'][0],
        num_blocks_reconstruction=params['nblocks'][1],
        center_frame_idx=1,  # invalid when TSA is off
        with_tsa=False),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='sum'))

train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=1)

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
    dict(type='Normalize',
         keys=['lq', 'gt'],
         mean=[0, 0, 0],
         std=[1, 1, 1],
         to_rgb=True),
    dict(type='PairedRandomCropQE',
         patch_size=params['patchsize'],
         keys=['lq', 'gt']),
    dict(type='Flip',
         keys=['lq', 'gt'],
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
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
    dict(type='Normalize',
         keys=['lq', 'gt'],
         mean=[0, 0, 0],
         std=[1, 1, 1],
         to_rgb=True),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'],
    )
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
                           folder=f'{dataset_lq_dir}',
                           gt_folder=f'{dataset_gt_dir}/sequences',
                           ann_file=f'{dataset_gt_dir}/tri_trainlist.txt',
                           pipeline=train_pipeline,
                           test_mode=False,
                           filename_tmpl='{}.png',
                           edge_padding=True,
                           center_gt=True)),
            val=dict(type=dataset_type,
                     folder=f'{dataset_lq_dir}',
                     gt_folder=f'{dataset_gt_dir}/sequences',
                     ann_file=f'{dataset_gt_dir}/tri_validlist.txt',
                     pipeline=test_pipeline,
                     test_mode=True,
                     filename_tmpl='{}.png',
                     edge_padding=True,
                     center_gt=True),
            test=dict(type=dataset_type,
                      folder=f'{dataset_lq_dir}',
                      gt_folder=f'{dataset_gt_dir}/sequences',
                      ann_file=f'{dataset_gt_dir}/tri_testlist.txt',
                      pipeline=test_pipeline,
                      test_mode=True,
                      filename_tmpl='{}.png',
                      edge_padding=True,
                      center_gt=True))

optimizers = dict(generator=dict(type='Adam', lr=4e-4, betas=(0.9, 0.999)))

total_iters = params['kiters'] * 1000
lr_config = dict(policy='CosineRestart',
                 by_epoch=False,
                 periods=[p * 1000 for p in params['klrperiods']],
                 restart_weights=[1, 0.5, 0.5, 0.5],
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
