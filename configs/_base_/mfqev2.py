train_pipeline = [
    dict(
        type="LoadImageFromFileList", io_backend="disk", key="lq", channel_order="rgb"
    ),
    dict(
        type="LoadImageFromFileList", io_backend="disk", key="gt", channel_order="rgb"
    ),
    dict(type="RescaleToZeroOne", keys=["lq", "gt"]),
    dict(type="PairedRandomCrop", gt_patch_size=256),  # keys must be 'lq' and 'gt'
    dict(type="Flip", keys=["lq", "gt"], flip_ratio=0.5, direction="horizontal"),
    dict(type="Flip", keys=["lq", "gt"], flip_ratio=0.5, direction="vertical"),
    dict(type="RandomTransposeHW", keys=["lq", "gt"], transpose_ratio=0.5),
    dict(type="FramesToTensor", keys=["lq", "gt"]),
    dict(type="Collect", keys=["lq", "gt"], meta_keys=["lq_path", "gt_path"]),
]
test_pipeline = [
    dict(
        type="LoadImageFromFileList", io_backend="disk", key="lq", channel_order="rgb"
    ),
    dict(
        type="LoadImageFromFileList", io_backend="disk", key="gt", channel_order="rgb"
    ),
    dict(type="RescaleToZeroOne", keys=["lq", "gt"]),
    dict(type="FramesToTensor", keys=["lq", "gt"]),
    dict(type="Collect", keys=["lq", "gt"], meta_keys=["lq_path", "gt_path", "key"]),
]

batchsize = 8
ngpus = 2
assert batchsize % ngpus == 0, (
    "Samples in a batch should better be evenly" " distributed among all GPUs."
)
batchsize_gpu = batchsize // ngpus

dataset_type = "PairedVideoDataset"

dataset_gt_root = "data/mfqev2"
dataset_lq_root = "data/mfqev2_lq/hm18.0/ldp/qp37"

data = dict(
    workers_per_gpu=batchsize_gpu,
    train_dataloader=dict(samples_per_gpu=batchsize_gpu, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type="RepeatDataset",
        times=1000,
        dataset=dict(
            type=dataset_type,
            lq_folder=f"{dataset_lq_root}/train",
            gt_folder=f"{dataset_gt_root}/train",
            pipeline=train_pipeline,
            ann_file="",
            test_mode=False,
            samp_len=-1,
            padding=False,
            center_gt=False,
        ),
    ),
    val=dict(
        type=dataset_type,
        lq_folder=f"{dataset_lq_root}/test",
        gt_folder=f"{dataset_gt_root}/test",
        pipeline=test_pipeline,
        ann_file="",
        test_mode=True,
        samp_len=-1,
        padding=True,
        center_gt=False,
    ),
    test=dict(
        type=dataset_type,
        lq_folder=f"{dataset_lq_root}/test",
        gt_folder=f"{dataset_gt_root}/test",
        pipeline=test_pipeline,
        ann_file="",
        test_mode=True,
        samp_len=-1,
        padding=True,
        center_gt=False,
    ),
)
