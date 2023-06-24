import os.path as osp

_base_ = ['flickr2k.py']

train_lq_folder = 'data/flickr2k_lq_lmdb/bpg/qp37/train.lmdb'
train_gt_folder = 'data/flickr2k_lmdb/train.lmdb'

# valid_lq_folder = 'data/flickr2k_lq_lmdb/bpg/qp37/test.lmdb'
# valid_gt_folder = 'data/flickr2k_lmdb/test.lmdb'

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
    # dict(type='PairedRandomCropQE', patch_size=128, keys=['lq', 'gt']),
    dict(type='Flip',
         keys=['lq', 'gt'],
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]
# test_pipeline = [
#     dict(type='LoadImageFromFile',
#          io_backend='lmdb',
#          db_path=valid_lq_folder,
#          key='lq',
#          channel_order='rgb'),
#     dict(type='LoadImageFromFile',
#          io_backend='lmdb',
#          db_path=valid_gt_folder,
#          key='gt',
#          channel_order='rgb'),
#     dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
#     dict(type='ImageToTensor', keys=['lq', 'gt']),
#     dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
# ]

# data = dict(train=dict(
#     dataset=dict(lq_folder=train_lq_folder,
#                  gt_folder=train_gt_folder,
#                  pipeline=train_pipeline,
#                  ann_file=osp.join(train_lq_folder, 'meta_info.txt'))),
#             val=dict(lq_folder=valid_lq_folder,
#                      gt_folder=valid_gt_folder,
#                      pipeline=test_pipeline,
#                      ann_file=osp.join(valid_lq_folder, 'meta_info.txt')),
#             test=dict(lq_folder=valid_lq_folder,
#                       gt_folder=valid_gt_folder,
#                       pipeline=test_pipeline,
#                       ann_file=osp.join(valid_lq_folder, 'meta_info.txt')))
data = dict(train=dict(
    dataset=dict(lq_folder=train_lq_folder,
                 gt_folder=train_gt_folder,
                 pipeline=train_pipeline,
                 ann_file=osp.join(train_lq_folder, 'meta_info.txt'))))
