_base_ = ['../_base_/runtime.py', '../_base_/vimeo90k_septuplet.py']

exp_name = 'stdf_vimeo90k_septuplet'

center_gt = True
model = dict(type='BasicVQERestorer',
             generator=dict(type='STDFNet',
                            io_channels=3,
                            radius=3,
                            nf_stdf=32,
                            nb_stdf=3,
                            nf_stdf_out=64,
                            nf_qe=48,
                            nb_qe=6),
             pixel_loss=dict(type='CharbonnierLoss',
                             loss_weight=1.0,
                             reduction='mean'),
             center_gt=center_gt)

data = dict(train=dict(dataset=dict(center_gt=center_gt)),
            val=dict(center_gt=center_gt),
            test=dict(center_gt=center_gt))

work_dir = f'work_dirs/{exp_name}'
