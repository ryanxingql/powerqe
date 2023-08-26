_base_ = ["../_base_/runtime.py", "../_base_/mfqev2_normalize.py"]

exp_name = "edvr_mfqev2"

center_gt = True
model = dict(
    type="BasicVQERestorer",
    generator=dict(
        type="EDVRNetQE",
        io_channels=3,
        mid_channels=32,
        num_frames=3,
        deform_groups=8,
        num_blocks_extraction=2,
        num_blocks_reconstruction=5,
        center_frame_idx=1,  # invalid when TSA is off
        with_tsa=False,
    ),
    pixel_loss=dict(type="CharbonnierLoss", loss_weight=1.0, reduction="mean"),
    center_gt=center_gt,
)

norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1])
test_cfg = dict(denormalize=norm_cfg)

data = dict(
    train=dict(dataset=dict(samp_len=3, center_gt=center_gt)),
    val=dict(samp_len=3, center_gt=center_gt),
    test=dict(samp_len=3, center_gt=center_gt),
)

work_dir = f"work_dirs/{exp_name}"
