default_scope = 'mmagic'
save_dir = 'work_dirs'

model = dict(type='BaseEditModel',
             pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
             train_cfg=dict(),
             test_cfg=dict(metrics=['PSNR'], crop_border=1),
             data_preprocessor=dict(
                 type='DataPreprocessor',
                 mean=[0., 0., 0.],
                 std=[255., 255., 255.],
             ))

val_evaluator = dict(type='Evaluator',
                     metrics=[
                         dict(type='MAE'),
                         dict(type='PSNR', crop_border=1),
                         dict(type='SSIM', crop_border=1)
                     ])

max_iters = 500 * 1000
train_cfg = dict(type='IterBasedTrainLoop',
                 max_iters=max_iters,
                 val_interval=5000)
val_cfg = dict(type='MultiValLoop')

optim_wrapper = dict(constructor='DefaultOptimWrapperConstructor',
                     type='OptimWrapper',
                     optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

param_scheduler = dict(type='CosineRestartLR',
                       by_epoch=False,
                       periods=[max_iters],
                       restart_weights=[1],
                       eta_min=1e-7)

custom_hooks = [dict(type='VisualizationHook')]
vis_backends = [dict(type='TensorboardVisBackend', save_dir=save_dir)]
visualizer = dict(type='Visualizer',
                  vis_backends=vis_backends,
                  save_dir=save_dir)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        out_dir=save_dir,
        by_epoch=False,
        max_keep_ckpts=10,
        save_best='PSNR',
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)

load_from = None
resume = False
