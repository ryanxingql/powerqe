_base_ = ['div2k.py']

train_lq_folder = 'data/div2k_lq_patches_lmdb/bpg/qp37/train.lmdb'
train_gt_folder = 'data/div2k_patches_lmdb/train.lmdb'

train_pipeline = [
    dict(type='LoadImageFromFile',
         io_backend='lmdb',
         db_path=train_lq_folder,
         key='lq',
         channel_order='rgb'),
    dict(type='LoadImageFromFile',
         io_backend='lmdb',
         db_path=train_gt_folder,
         key='gt',
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    # no need to crop
    dict(type='Flip',
         keys=['lq', 'gt'],
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

data = dict(train=dict(dataset=dict(type='SRLmdbDataset',
                                    lq_folder=train_lq_folder,
                                    gt_folder=train_gt_folder,
                                    pipeline=train_pipeline,
                                    scale=1)))
