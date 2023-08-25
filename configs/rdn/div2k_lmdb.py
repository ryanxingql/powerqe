_base_ = ["../_base_/runtime.py", "../_base_/div2k_lmdb.py"]

exp_name = "rdn_div2k"

model = dict(
    type="BasicQERestorer",
    generator=dict(
        type="RDNQE", rescale=1, io_channels=3, mid_channels=32, num_blocks=4
    ),
    pixel_loss=dict(type="L1Loss", loss_weight=1.0, reduction="mean"),
)

work_dir = f"work_dirs/{exp_name}"
