_base_ = ['../_base_/runtime.py', '../_base_/div2k_lmdb.py']

exp_name = 'arcnn_div2k'

model = dict(type='BasicQERestorer',
             generator=dict(type='ARCNN',
                            io_channels=3,
                            mid_channels_1=64,
                            mid_channels_2=32,
                            mid_channels_3=16,
                            in_kernel_size=9,
                            mid_kernel_size_1=7,
                            mid_kernel_size_2=1,
                            out_kernel_size=5),
             pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))

work_dir = f'work_dirs/{exp_name}'
