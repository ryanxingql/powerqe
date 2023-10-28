_base_ = ["../../_base_/runtime.py", "../../_base_/bpg/div2k_qp32_lmdb.py"]

exp_name = "dcad_div2k_qp32"

model = dict(
    type="BasicQERestorer",
    generator=dict(type="DCAD", io_channels=3, mid_channels=64, num_blocks=8),
    pixel_loss=dict(type="L1Loss", loss_weight=1.0, reduction="mean"),
)

work_dir = f"work_dirs/{exp_name}"
