_base_ = 'provqe_vimeo90k_triplet.py'

exp_name = 'provqe_vimeo90k_septuplet'

batchsize = 8
ngpus = 2
assert batchsize % ngpus == 0, ('Samples in a batch should better be evenly'
                                ' distributed among all GPUs.')
dataset_gt_dir = 'data/vimeo_septuplet'
dataset_lq_dir = 'data/vimeo_septuplet_lq'
key_frames = [1, 0, 1, 0, 1, 0, 1]
batchsize_gpu = batchsize // ngpus
data = dict(
    workers_per_gpu=batchsize_gpu,
    train_dataloader=dict(samples_per_gpu=batchsize_gpu),
    train=dict(dataset=dict(key_frames=key_frames,
                            lq_folder=f'{dataset_lq_dir}',
                            gt_folder=f'{dataset_gt_dir}/sequences',
                            ann_file=f'{dataset_gt_dir}/sep_trainlist.txt')),
    val=dict(key_frames=key_frames,
             lq_folder=f'{dataset_lq_dir}',
             gt_folder=f'{dataset_gt_dir}/sequences',
             ann_file=f'{dataset_gt_dir}/sep_validlist.txt'),
    test=dict(key_frames=key_frames,
              lq_folder=f'{dataset_lq_dir}',
              gt_folder=f'{dataset_gt_dir}/sequences',
              ann_file=f'{dataset_gt_dir}/sep_testlist.txt'))

work_dir = f'work_dirs/{exp_name}'
