_base_ = ["../_base_/runtime.py", "../_base_/div2k.py"]

exp_name = "unet_div2k"

model = dict(
    type="BasicQERestorer",
    generator=dict(
        type="UNet",
        nf_in=3,
        nf_out=3,
        nlevel=3,
        nf_base=64,
        nf_gr=2,
        nl_gr=2,
        nf_max=1024,
        nl_base=1,
        nl_max=8,
        down="avepool2d",
        up="transpose2d",
        reduce="concat",
        residual=True,
    ),
    pixel_loss=dict(type="L1Loss", loss_weight=1.0, reduction="mean"),
)

work_dir = f"work_dirs/{exp_name}"
