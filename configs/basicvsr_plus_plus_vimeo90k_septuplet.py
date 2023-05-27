_base_ = 'basicvsr_plus_plus_vimeo90k_triplet.py'

exp_name = 'basicvsr_plus_plus_vimeo90k_septuplet'

batchsize = 8
ngpus = 2
assert batchsize % ngpus == 0, ('Samples in a batch should better be evenly'
                                ' distributed among all GPUs.')
dataset_gt_root = 'data/vimeo_septuplet'
dataset_lq_folder = 'data/vimeo_septuplet_lq'
batchsize_gpu = batchsize // ngpus
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
