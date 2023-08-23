_base_ = ["../_base_/runtime.py", "../_base_/mfqev2.py"]

exp_name = "basicvsr_plus_plus_mfqev2"

center_gt = False
model = dict(
    type="BasicVQERestorer",
    generator=dict(
        type="BasicVSRPlusPlus",
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=False,
        spynet_pretrained="https://download.openmmlab.com/mmediting/restorers/"
        "basicvsr/spynet_20210409-c6c1bd09.pth",
    ),
    pixel_loss=dict(type="CharbonnierLoss", loss_weight=1.0, reduction="mean"),
    center_gt=center_gt,
)

train_cfg = dict(
    _delete_=True, fix_iter=5000, fix_module=["edvr", "spynet"]
)  # set "_delete_=True" to replace None
test_cfg = dict(padding=dict(minSize=256))

data = dict(
    train=dict(dataset=dict(samp_len=7, center_gt=center_gt)),
    val=dict(samp_len=7, stride=7, padding=False, center_gt=center_gt),
    test=dict(samp_len=7, stride=7, padding=False, center_gt=center_gt),
)

optimizers = dict(
    generator=dict(paramwise_cfg=dict(custom_keys={"spynet": dict(lr_mult=0.25)}))
)

work_dir = f"work_dirs/{exp_name}"
find_unused_parameters = True  # for spynet pre-training
