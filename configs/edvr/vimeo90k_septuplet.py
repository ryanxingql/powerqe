_base_ = ["../_base_/runtime.py", "../_base_/vimeo90k_septuplet_normalize.py"]

exp_name = "edvr_vimeo90k_septuplet"

center_gt = True
model = dict(
    type="BasicVQERestorer",
    generator=dict(
        type="EDVRNetQE",
        io_channels=3,
        mid_channels=64,
        num_frames=7,
        deform_groups=8,
        num_blocks_extraction=5,
        num_blocks_reconstruction=10,
        center_frame_idx=1,  # invalid when TSA is off
        with_tsa=False,
    ),
    pixel_loss=dict(type="CharbonnierLoss", loss_weight=1.0, reduction="mean"),
    center_gt=center_gt,
)

norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1])
test_cfg = dict(denormalize=norm_cfg)

data = dict(
    train=dict(dataset=dict(center_gt=center_gt)),
    val=dict(center_gt=center_gt),
    test=dict(center_gt=center_gt),
)

work_dir = f"work_dirs/{exp_name}"
