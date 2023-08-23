_base_ = ["div2k_stage1.py"]

exp_name = "esrgan_div2k_stage2"

model = dict(
    _delete_=True,
    type="ESRGANRestorer",
    generator=dict(
        type="RRDBNetQE",
        io_channels=3,
        mid_channels=64,
        num_blocks=23,
        growth_channels=32,
        upscale_factor=1,
    ),
    discriminator=dict(type="ModifiedVGG", in_channels=3, mid_channels=64),
    pixel_loss=dict(type="L1Loss", loss_weight=1e-2, reduction="mean"),
    perceptual_loss=dict(
        type="PerceptualLoss",
        layer_weights={"34": 1.0},
        vgg_type="vgg19",
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False,
        pretrained="https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    ),
    gan_loss=dict(
        type="GANLoss",
        gan_type="vanilla",
        loss_weight=5e-3,
        real_label_val=1.0,
        fake_label_val=0,
    ),
    pretrained="work_dirs/esrgan_div2k_stage1/latest.pth",
)

optimizers = dict(discriminator=dict(type="Adam", lr=1e-4, betas=(0.9, 0.999)))

total_iters = 400 * 1000
lr_config = dict(
    _delete_=True,
    policy="Step",
    by_epoch=False,
    step=[s * 1000 for s in [50, 100, 200, 300]],
    gamma=0.5,
)

val_inter = total_iters // 10
checkpoint_config = dict(interval=val_inter)
evaluation = dict(interval=val_inter)

work_dir = f"work_dirs/{exp_name}"
