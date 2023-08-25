_base_ = ["../_base_/runtime.py", "../_base_/div2k_lmdb.py"]

exp_name = "mprnet_div2k"

model = dict(
    type="BasicQERestorer",
    generator=dict(type="MPRNet", io_c=3, n_feat=96),
    pixel_loss=dict(type="CharbonnierLoss", loss_weight=1.0, reduction="mean"),
)

test_cfg = dict(unfolding=dict(patchsize=128, splits=4))  # to save memory

work_dir = f"work_dirs/{exp_name}"
