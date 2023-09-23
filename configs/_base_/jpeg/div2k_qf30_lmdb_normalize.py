_base_ = "div2k_qf30_normalize.py"

norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1])

train_lq_folder = "data/lmdb/div2k_lq/jpeg/qf30/train.lmdb"
train_gt_folder = "data/lmdb/div2k/train.lmdb"

train_pipeline = [
    dict(
        type="LoadImageFromFile",
        io_backend="lmdb",
        db_path=train_lq_folder,
        key="lq",
        channel_order="rgb",
    ),
    dict(
        type="LoadImageFromFile",
        io_backend="lmdb",
        db_path=train_gt_folder,
        key="gt",
        channel_order="rgb",
    ),
    dict(type="RescaleToZeroOne", keys=["lq", "gt"]),
    dict(type="Normalize", keys=["lq", "gt"], **norm_cfg),
    # no need to crop
    dict(type="Flip", keys=["lq", "gt"], flip_ratio=0.5, direction="horizontal"),
    dict(type="Flip", keys=["lq", "gt"], flip_ratio=0.5, direction="vertical"),
    dict(type="RandomTransposeHW", keys=["lq", "gt"], transpose_ratio=0.5),
    dict(type="ImageToTensor", keys=["lq", "gt"]),
    dict(type="Collect", keys=["lq", "gt"], meta_keys=["lq_path", "gt_path"]),
]

data = dict(
    train=dict(
        dataset=dict(
            type="SRLmdbDataset",
            lq_folder=train_lq_folder,
            gt_folder=train_gt_folder,
            pipeline=train_pipeline,
            scale=1,
        )
    )
)
