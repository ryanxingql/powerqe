_base_ = ['arcnn_div2k.py']

train_lq_folder = 'data/div2k_lq_lmdb/bpg/qp37/train.lmdb'
train_gt_folder = 'data/div2k_lmdb/train.lmdb'
valid_lq_folder = 'data/div2k_lq_lmdb/bpg/qp37/valid.lmdb'
valid_gt_folder = 'data/div2k_lmdb/valid.lmdb'

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
    # no cropping
    dict(type='Flip',
         keys=['lq', 'gt'],
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]
test_pipeline = [
    dict(type='LoadImageFromFile',
         io_backend='lmdb',
         db_path=valid_lq_folder,
         key='lq',
         channel_order='rgb'),
    dict(type='LoadImageFromFile',
         io_backend='lmdb',
         db_path=valid_gt_folder,
         key='gt',
         channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

dataset_type = 'SRLmdbDataset'
data = dict(train=dict(dataset=dict(_delete_=True,
                                    type=dataset_type,
                                    lq_folder=train_lq_folder,
                                    gt_folder=train_gt_folder,
                                    pipeline=train_pipeline,
                                    scale=1,
                                    test_mode=False)),
            val=dict(_delete_=True,
                     type=dataset_type,
                     lq_folder=valid_lq_folder,
                     gt_folder=valid_gt_folder,
                     pipeline=test_pipeline,
                     scale=1,
                     test_mode=True),
            test=dict(_delete_=True,
                      type=dataset_type,
                      lq_folder=valid_lq_folder,
                      gt_folder=valid_gt_folder,
                      pipeline=test_pipeline,
                      scale=1,
                      test_mode=True))
