_base_ = ['_base_/runtime.py', '_base_/datasets/div2k.py']

experiment_name = 'arcnn_div2k'
work_dir = f'work_dirs/{experiment_name}'

custom_imports = dict(imports=['powerqe.models.backbones.arcnn'],
                      allow_failed_imports=False)

model = dict(generator=dict(type='ARCNN',
                            io_channels=3,
                            mid_channels_1=64,
                            mid_channels_2=32,
                            mid_channels_3=16,
                            in_kernel_size=9,
                            mid_kernel_size_1=7,
                            mid_kernel_size_2=1,
                            out_kernel_size=5))
