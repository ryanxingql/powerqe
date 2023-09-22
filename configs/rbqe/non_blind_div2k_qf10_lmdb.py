_base_ = ["../_base_/runtime.py", "../_base_/div2k_qf10_lmdb.py"]

exp_name = "rbqe_non_blind_div2k_qf10"

model = dict(
    type="BasicQERestorer",
    generator=dict(
        type="RBQE",
        nf_io=3,
        nf_base=32,
        nlevel=5,
        down_method="strideconv",
        up_method="transpose2d",
        if_separable=False,
        if_eca=False,
        if_only_last_output=True,  # non-blind
        comp_type="hevc",
    ),
    pixel_loss=dict(type="L1Loss", loss_weight=1.0, reduction="mean"),
)

work_dir = f"work_dirs/{exp_name}"
