_base_ = ['arcnn_div2k.py']

exp_name = 'debug'

val_inter = 100
checkpoint_config = dict(interval=val_inter,
                         save_optimizer=True,
                         by_epoch=False)
evaluation = dict(interval=val_inter)

work_dir = f'work_dirs/{exp_name}'
