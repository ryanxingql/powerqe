_base_ = ['base.py']

train_dataloader = dict(dataset=dict(data_root='data/div2k/train',
                                     data_prefix=dict(img='lq', gt='gt')))
val_dataloader = dict(dataset=dict(data_root='data/div2k/valid',
                                   data_prefix=dict(img='lq', gt='gt')))
