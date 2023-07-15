_base_ = ['_base_/runtime.py', '_base_/mfqev2.py']

exp_name = 'stdf_mfqev2'

center_gt = True
model = dict(type='BasicVQERestorer',
             generator=dict(type='STDFNet',
                            io_channels=3,
                            radius=1,
                            nf_stdf=32,
                            nb_stdf=3,
                            nf_stdf_out=64,
                            nf_qe=48,
                            nb_qe=6),
             pixel_loss=dict(type='CharbonnierLoss',
                             loss_weight=1.0,
                             reduction='mean'),
             center_gt=center_gt)

data = dict(train=dict(dataset=dict(center_gt=center_gt, samp_len=3)),
            val=dict(center_gt=center_gt, samp_len=3),
            test=dict(center_gt=center_gt, samp_len=3))

work_dir = f'work_dirs/{exp_name}'
