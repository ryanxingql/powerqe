# RyanXingQL @2022
import math
import os.path as osp
from collections import defaultdict

import numpy as np
from mmedit.core.registry import build_metric
from mmedit.datasets import SRFolderDataset

from .pipelines.compose import Compose
from .registry import DATASETS

FEATURE_BASED_METRICS = ['FID', 'KID']


@DATASETS.register_module()
class PairedSameSizeImageDataset(SRFolderDataset):
    """Paired image dataset. GT and LQ are with the same size.

    Differences to SRFolderDataset:
        Scale is set to 1.
        Use the Compose in PowerQE.

    Args:
        lq_folder (str | :obj:Path): LQ folder.
        gt_folder (str | :obj:Path): GT folder.
        pipeline (List[dict | callable]): A sequence of data transformations.
        ann_file (str | :obj:Path): Path to the annotation file.
            Each line records an image path relative to the GT/LQ folder.
        test_mode (bool): Store True when building test dataset.
            Default: False.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 pipeline,
                 ann_file='',
                 test_mode=False):
        self.ann_file = ann_file
        # BaseDataset cannot accept any pipelines outside MMEditing
        # pass [] into __init__
        super().__init__(lq_folder=lq_folder,
                         gt_folder=gt_folder,
                         pipeline=[],
                         scale=1,
                         test_mode=test_mode)
        self.pipeline = Compose(pipeline)

    def load_annotations(self):
        """Scan GT and LQ folders and record samples.

        The GT folder includes all images by default.
        The GT and LQ images have the same name (also extension) by default.
            If different, please use an annotation file.

        Returns:
            list[dict]: Sample information.
        """
        data_infos = []

        if self.ann_file:
            with open(self.ann_file, 'r') as f:
                img_names = f.read().split('\n')
            img_names = [
                n.strip() for n in img_names
                if (n.strip() is not None and n != '')
            ]
            gt_paths = [osp.join(self.gt_folder, name) for name in img_names]
            lq_paths = [osp.join(self.lq_folder, name) for name in img_names]

            for gt_path, lq_path in zip(gt_paths, lq_paths):
                data_infos.append(dict(gt_path=gt_path, lq_path=lq_path))
        else:
            gt_paths = self.scan_folder(self.gt_folder)
            lq_paths = self.scan_folder(self.lq_folder)
            if len(gt_paths) != len(lq_paths):
                raise ValueError(
                    'GT and LQ folders should have the same number of images;'
                    f' found {len(gt_paths)} and {len(lq_paths)} images,'
                    ' respectively.')

            for gt_path in gt_paths:
                basename, ext = osp.splitext(osp.basename(gt_path))
                lq_path = osp.join(self.lq_folder, basename + ext)
                if lq_path not in lq_paths:
                    raise FileNotFoundError(
                        f'Cannot find "{lq_path}" in "{self.lq_folder}".')
                data_infos.append(dict(gt_path=gt_path, lq_path=lq_path))

        return data_infos

    def evaluate2(self, results, logger=None):
        """Evaluate with different metrics.

        Difference to that of BaseSRDataset: Deal with inf PSNR values.

        Args:
            results (list[tuple]): The output of forward_test() of the model.
                results: [item1,item2, ... , itemN]; N is the sample number.
                item: dict(eval_result=dict(
                    metric1=value1,
                    metric2=value2,
                    ...),
                ...)

        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        # Collect eval values of samples under each metric into a list
        # and record lists into a dict where keys are metric names

        results = [res['eval_result'] for res in results]  # a list of dict
        eval_result = defaultdict(list)  # a dict of list
        for res in results:
            for metric, val in res.items():
                eval_result[metric].append(val)

        # Deal with inf PSNR

        for metric, values in eval_result.items():
            assert len(values) == len(self), (
                f'Length of evaluation result of {metric} is {len(values)}, '
                f'should be {len(self)}')

            if 'PSNR' in metric:
                values_pro = [v for v in values if not math.isinf(v)]
                if len(values_pro) < len(values):
                    ndel = len(values) - len(values_pro)
                    if ndel == 1:
                        print('Ignore a PSNR result of inf dB.')
                    else:
                        print(f'Ignore {ndel} PSNR results of inf dB.')
                    eval_result[metric] = values_pro

        # Replace lists with average scores

        eval_result.update({
            metric: sum(values) / len(self)
            for metric, values in eval_result.items()
            if metric not in ['_inception_feat'] + FEATURE_BASED_METRICS
        })

        # Evaluate feature-based metrics

        if '_inception_feat' in eval_result:
            feat1, feat2 = [], []
            for f1, f2 in eval_result['_inception_feat']:
                feat1.append(f1)
                feat2.append(f2)
            feat1 = np.concatenate(feat1, 0)
            feat2 = np.concatenate(feat2, 0)

            for metric in FEATURE_BASED_METRICS:
                if metric in eval_result:
                    metric_func = build_metric(eval_result[metric].pop())
                    eval_result[metric] = metric_func(feat1, feat2)

            # delete a redundant key for clean logging
            del eval_result['_inception_feat']

        return eval_result
