_base_ = [
    '_base_/runtime.py', '_base_/vimeo90k_septuplet.py',
    'basicvsr_plus_plus_vimeo90k_septuplet.py'
]

exp_name = 'provqe_vimeo90k_septuplet'

model = dict(type='ProVQERestorer', generator=dict(type='ProVQE'))

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
         meta_keys=['lq_path', 'gt_path', 'key_frms'])
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
         meta_keys=['lq_path', 'gt_path', 'key', 'key_frms'])
]

dataset_type = 'PairedSameSizeVideoKeyAnnotationsDataset'
key_frames = [1, 0, 1, 0, 1, 0, 1]
data = dict(train=dict(dataset=dict(
    type=dataset_type, pipeline=train_pipeline, key_frames=key_frames)),
            val=dict(type=dataset_type,
                     pipeline=test_pipeline,
                     key_frames=key_frames),
            test=dict(type=dataset_type,
                      pipeline=test_pipeline,
                      key_frames=key_frames))

work_dir = f'work_dirs/{exp_name}'
