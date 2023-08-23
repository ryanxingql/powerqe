_base_ = ["../_base_/runtime.py", "../_base_/vimeo90k_triplet.py"]

exp_name = "mfqev2_vimeo90k_triplet"

center_gt = True
model = dict(
    type="BasicVQERestorer",
    generator=dict(
        type="MFQEv2",
        io_channels=3,
        nf=32,
        spynet_pretrained="https://download.openmmlab.com/mmediting/restorers/"
        "basicvsr/spynet_20210409-c6c1bd09.pth",
    ),
    pixel_loss=dict(type="CharbonnierLoss", loss_weight=1.0, reduction="mean"),
    center_gt=center_gt,
)

train_cfg = dict(
    _delete_=True, fix_iter=5000, fix_module=["spynet"]
)  # set "_delete_=True" to replace None

dataset_type = "PairedVideoKeyFramesDataset"
key_frames = [1, 0, 1]

data = dict(
    train=dict(
        dataset=dict(type=dataset_type, center_gt=center_gt, key_frames=key_frames)
    ),
    val=dict(type=dataset_type, center_gt=center_gt, key_frames=key_frames),
    test=dict(type=dataset_type, center_gt=center_gt, key_frames=key_frames),
)

work_dir = f"work_dirs/{exp_name}"
find_unused_parameters = True  # for spynet pre-training
