_base_ = 'edvr_vimeo90k_triplet.py'

exp_name = 'edvr_vimeo90k_septuplet'

model = dict(generator=dict(num_frames=7))

batchsize = 8
ngpus = 2
assert batchsize % ngpus == 0, ('Samples in a batch should better be evenly'
                                ' distributed among all GPUs.')
dataset_gt_root = 'data/vimeo_septuplet'
dataset_lq_folder = 'data/vimeo_septuplet_lq'
batchsize_gpu = batchsize // ngpus
# since there are only three frames in a sequence
# two of which need padding in testing
# training also use padding
data = dict(
    workers_per_gpu=batchsize_gpu,
    train_dataloader=dict(samples_per_gpu=batchsize_gpu),
    train=dict(dataset=dict(lq_folder=f'{dataset_lq_folder}',
                            gt_folder=f'{dataset_gt_root}/sequences',
                            ann_file=f'{dataset_gt_root}/sep_trainlist.txt')),
    val=dict(lq_folder=f'{dataset_lq_folder}',
             gt_folder=f'{dataset_gt_root}/sequences',
             ann_file=f'{dataset_gt_root}/sep_validlist.txt'),
    test=dict(lq_folder=f'{dataset_lq_folder}',
              gt_folder=f'{dataset_gt_root}/sequences',
              ann_file=f'{dataset_gt_root}/sep_testlist.txt'))

work_dir = f'work_dirs/{exp_name}'
