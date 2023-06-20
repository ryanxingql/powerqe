train_pipeline = [
    dict(type='LoadImageFromFileMultiKeys',
         io_backend='disk',
         keys=['lq', 'gt'],
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCropQE', patch_size=128, keys=['lq', 'gt']),
    dict(type='Flip',
         keys=['lq', 'gt'],
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
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

batchsize = 32
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
                       dataset=dict(
                           type=dataset_type,
                           lq_folder='data/flickr2k_lq/bpg/qp37',
                           gt_folder='data/flickr2k',
                           pipeline=train_pipeline,
                           ann_file='data/flickr2k_lq/bpg/qp37/train.txt',
                           test_mode=False)),
            val=dict(type=dataset_type,
                     lq_folder='data/flickr2k_lq/bpg/qp37',
                     gt_folder='data/flickr2k',
                     pipeline=test_pipeline,
                     ann_file='data/flickr2k_lq/bpg/qp37/test.txt',
                     test_mode=True),
            test=dict(type=dataset_type,
                      lq_folder='data/flickr2k_lq/bpg/qp37',
                      gt_folder='data/flickr2k',
                      pipeline=test_pipeline,
                      ann_file='data/flickr2k_lq/bpg/qp37/test.txt',
                      test_mode=True))
