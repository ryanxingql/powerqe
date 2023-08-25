_base_ = ["../_base_/runtime.py", "../_base_/div2k_lmdb.py"]

exp_name = "cbdnet_div2k"

model = dict(
    type="BasicQERestorer",
    generator=dict(
        type="CBDNet",
        io_channels=3,
        estimate_channels=32,
        nlevel_denoise=3,
        nf_base_denoise=64,
    ),
    pixel_loss=dict(type="L1Loss", loss_weight=1.0, reduction="mean"),
)

work_dir = f"work_dirs/{exp_name}"
