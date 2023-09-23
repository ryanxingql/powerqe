_base_ = ["../../_base_/runtime.py", "../../_base_/div2k_qf10_lmdb_normalize.py"]

exp_name = "esrgan_div2k_qf10_stage1"
work_dir = f"work_dirs/{exp_name}"

model = dict(
    type="BasicQERestorer",
    generator=dict(
        type="RRDBNetQE",
        io_channels=3,
        mid_channels=32,
        num_blocks=16,
        growth_channels=32,
        upscale_factor=1,
    ),
    pixel_loss=dict(type="L1Loss", loss_weight=1.0, reduction="mean"),
)

norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1])
test_cfg = dict(denormalize=norm_cfg)
