_base_ = ["../../_base_/runtime.py", "../../_base_/bpg/div2k_qp42_lmdb.py"]

exp_name = "mprnet_div2k_qp42"

model = dict(
    type="BasicQERestorer",
    generator=dict(
        type="MPRNet",
        io_c=3,
        n_feat=16,
        scale_unetfeats=16,
        scale_orsnetfeats=16,
        num_cab=4,
        reduction=4,
    ),
    pixel_loss=dict(type="CharbonnierLoss", loss_weight=1.0, reduction="mean"),
)

test_cfg = dict(unfolding=dict(patchsize=128, splits=4))  # to save memory

work_dir = f"work_dirs/{exp_name}"
