norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1])
train_pipeline = [
    dict(type="LoadImageFromFile", io_backend="disk", key="lq", channel_order="rgb"),
    dict(type="LoadImageFromFile", io_backend="disk", key="gt", channel_order="rgb"),
    dict(type="RescaleToZeroOne", keys=["lq", "gt"]),
    dict(type="Normalize", keys=["lq", "gt"], **norm_cfg),
    dict(type="PairedRandomCrop", gt_patch_size=128),  # keys must be 'lq' and 'gt'
    dict(type="Flip", keys=["lq", "gt"], flip_ratio=0.5, direction="horizontal"),
    dict(type="Flip", keys=["lq", "gt"], flip_ratio=0.5, direction="vertical"),
    dict(type="RandomTransposeHW", keys=["lq", "gt"], transpose_ratio=0.5),
    dict(type="ImageToTensor", keys=["lq", "gt"]),
    dict(type="Collect", keys=["lq", "gt"], meta_keys=["lq_path", "gt_path"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile", io_backend="disk", key="lq", channel_order="rgb"),
    dict(type="LoadImageFromFile", io_backend="disk", key="gt", channel_order="rgb"),
    dict(type="RescaleToZeroOne", keys=["lq", "gt"]),
    dict(type="Normalize", keys=["lq", "gt"], **norm_cfg),
    dict(type="ImageToTensor", keys=["lq", "gt"]),
    dict(type="Collect", keys=["lq", "gt"], meta_keys=["lq_path", "gt_path"]),
]

batchsize = 32
ngpus = 1
assert batchsize % ngpus == 0, (
    "Samples in a batch should better be evenly" " distributed among all GPUs."
)
batchsize_gpu = batchsize // ngpus

dataset_type = "SRFolderDataset"
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
            lq_folder="data/div2k_lq/bpg/qp27/train",
            gt_folder="data/div2k/train",
            pipeline=train_pipeline,
            scale=1,
            test_mode=False,
        ),
    ),
    val=dict(
        type=dataset_type,
        lq_folder="data/div2k_lq/bpg/qp27/valid",
        gt_folder="data/div2k/valid",
        pipeline=test_pipeline,
        scale=1,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        lq_folder="data/div2k_lq/bpg/qp27/valid",
        gt_folder="data/div2k/valid",
        pipeline=test_pipeline,
        scale=1,
        test_mode=True,
    ),
)
