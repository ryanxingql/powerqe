batchsize = 32
ngpus = 1
assert batchsize % ngpus == 0, ('Samples in a batch should better be evenly'
                                ' distributed among all GPUs.')
batchsize_gpu = batchsize // ngpus

dataset_type = 'BasicImageDataset'

train_pipeline = [
    dict(type='LoadImageFromFile',
         key='img',
         color_type='color',
         channel_order='rgb',
         imdecode_backend='pillow'),
    dict(type='LoadImageFromFile',
         key='gt',
         color_type='color',
         channel_order='rgb',
         imdecode_backend='pillow'),
    # transform `PairedRandomCrop` requires `results['scale']`
    dict(type='SetValues', dictionary=dict(scale=1)),
    dict(type='PairedRandomCrop', gt_patch_size=128, lq_key='img',
         gt_key='gt'),
    dict(type='Flip',
         keys=['img', 'gt'],
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5,
         direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='PackInputs', keys=['img', 'gt'])
]

val_pipeline = [
    dict(type='LoadImageFromFile',
         key='img',
         color_type='color',
         channel_order='rgb',
         imdecode_backend='pillow'),
    dict(type='LoadImageFromFile',
         key='gt',
         color_type='color',
         channel_order='rgb',
         imdecode_backend='pillow'),
    dict(type='PackInputs', keys=['img', 'gt'])
]

train_dataloader = dict(
    num_workers=batchsize_gpu,
    batch_size=batchsize_gpu,
    # pin_memory=True,
    drop_last=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type=dataset_type, pipeline=train_pipeline))
val_dataloader = dict(
    num_workers=batchsize_gpu,
    # pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, pipeline=val_pipeline))
