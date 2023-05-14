_base_ = 'stdf_vimeo90k_triplet.py'

# use center cropping to save memory
test_pipeline = [
    dict(type='LoadImageFromFileMultiKeys',
         io_backend='disk',
         keys=['lq', 'gt'],
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedCenterCrop', patch_size=128, keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect',
         keys=['lq', 'gt'],
         meta_keys=['lq_path', 'gt_path', 'key'])
]

# use small batchsize to save memory
# use new pipeline
batchsize = 2
ngpus = 1
assert batchsize % ngpus == 0, ('Samples in a batch should better be evenly'
                                ' distributed among all GPUs.')
batchsize_gpu = batchsize // ngpus
data = dict(workers_per_gpu=batchsize_gpu,
            train_dataloader=dict(samples_per_gpu=batchsize_gpu),
            val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))

# use small interval to early evaluation
# remove gpu_collect when using cpu
evaluation = dict(_delete_=True,
                  interval=110,
                  save_image=False,
                  gpu_collect=True)
