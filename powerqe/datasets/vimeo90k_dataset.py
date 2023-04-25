# RyanXingQL @2023
import copy
import os
import os.path as osp

from mmedit.datasets import BaseVFIDataset

from .pipelines.compose import Compose
from .registry import DATASETS


@DATASETS.register_module()
class Vimeo90KTripletCenterGTDataset(BaseVFIDataset):
    """
    Difference to the VFIVimeo90KDataset in mmedit:
        1. Require GT for im2.png.
        2. Support different ext between GT and LQ
            with filename_tmpl.
        3. Use the Compose in powerqe.

    New args:
        gt_folder: for GT im2.
        filename_tmpl (str): Template for each filename of LQ.
            Default: '{}.png'.
    """

    def __init__(self,
                 pipeline,
                 folder,
                 gt_folder,
                 ann_file,
                 test_mode=False,
                 filename_tmpl='{}.png'):
        super().__init__(pipeline, folder, ann_file, test_mode)

        self.lq_folder = self.folder
        self.gt_folder = str(gt_folder)
        self.filename_tmpl = filename_tmpl
        self.data_infos = self.load_annotations()

        # BaseDataset cannot accept the new pipeline outside MMEdit
        # we have to create pipeline manually
        self.pipeline = Compose(pipeline)

    def __getitem__(self, idx):
        """
        Difference to that of BaseVFIDataset:
            1. Add results['scale'] = 1 for PairedRandomCrop.
        """
        results = copy.deepcopy(self.data_infos[idx])
        results['folder'] = self.folder
        results['ann_file'] = self.ann_file
        results['scale'] = 1
        return self.pipeline(results)

    def load_annotations(self):
        """
        Difference to that of VFIVimeo90KDataset:
            1. Load LQ and GT for im2.
            2. Record 3 sub-sequences for one sequence.
        """
        # get keys
        with open(self.ann_file, 'r') as f:
            keys = f.read().split('\n')
            keys = [
                k.strip() for k in keys if (k.strip() is not None and k != '')
            ]

        data_infos = []
        for key in keys:
            key = key.replace('/', os.sep)
            key_folder = osp.join(self.lq_folder, key)

            # GT: im1
            lq_path = [
                osp.join(key_folder, f'{self.filename_tmpl.format("im1")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im1")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im2")}')
            ]
            gt_path = osp.join(self.gt_folder, key, 'im1.png')
            data_infos.append(
                dict(
                    lq_path=lq_path,
                    gt_path=gt_path,
                    key=key + '/im1',
                ))

            # GT: im2
            lq_path = [
                osp.join(key_folder, f'{self.filename_tmpl.format("im1")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im2")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im3")}')
            ]
            gt_path = osp.join(self.gt_folder, key, 'im2.png')
            data_infos.append(
                dict(
                    lq_path=lq_path,
                    gt_path=gt_path,
                    key=key + '/im2',
                ))

            # GT: im3
            lq_path = [
                osp.join(key_folder, f'{self.filename_tmpl.format("im2")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im3")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im3")}')
            ]
            gt_path = osp.join(self.gt_folder, key, 'im3.png')
            data_infos.append(
                dict(
                    lq_path=lq_path,
                    gt_path=gt_path,
                    key=key + '/im3',
                ))

        return data_infos


@DATASETS.register_module()
class CompressedVimeo90KTripletCenterGTDataset(Vimeo90KTripletCenterGTDataset):
    """
    Difference to Vimeo90KTripletCenterGTDataset:
        1. Use high-quality frames instead of neighboring frames.

    New args: qp_info (dict). See PowerQE doc.
    """

    def __init__(
        self,
        pipeline,
        folder,
        gt_folder,
        ann_file,
        test_mode=False,
        filename_tmpl='{}.png',
        qp_info=dict(
            qp=37,
            intra_qp_offset=-1,
            qp_offset=[5, 4] * 3 + [5, 1],
            qp_offset_model_off=[-6.5] * 7 + [0],
            qp_offset_model_scale=[0.2590] * 7 + [0],
        ),
    ):
        self.qps = self.cal_qps(qp_info, nfrms=3)
        super().__init__(
            pipeline=pipeline,
            folder=folder,
            gt_folder=gt_folder,
            ann_file=ann_file,
            test_mode=test_mode,
            filename_tmpl=filename_tmpl,
        )

    @staticmethod
    def cal_qps(qp_info, nfrms=3):
        # POC 0
        qps = [qp_info['qp'] + qp_info['intra_qp_offset']]

        # POC 1, 2, ...
        for idx in range(nfrms - 1):
            idx1 = idx % len(qp_info['qp_offset'])
            qp = qp_info['qp'] + qp_info['qp_offset'][idx1] +\
                qp_info['qp_offset_model_off'][idx1] +\
                qp_info['qp_offset_model_scale'][idx1] * qp_info['qp']
            qps.append(qp)
        return qps

    def load_annotations(self):
        """
        Difference to that of Vimeo90KTripletCenterGTDataset:
            1. Record high-quality frames instead of neighboring frames.
        """
        # get keys
        with open(self.ann_file, 'r') as f:
            keys = f.read().split('\n')
            keys = [
                k.strip() for k in keys if (k.strip() is not None and k != '')
            ]

        # find key frames
        if_1g2 = True if self.qps[0] < self.qps[1] else False
        if_2g3 = True if self.qps[1] < self.qps[2] else False
        im1_right = 'im2' if if_2g3 else 'im3'
        im3_left = 'im1' if if_1g2 else 'im2'

        data_infos = []
        for key in keys:
            key = key.replace('/', os.sep)
            key_folder = osp.join(self.lq_folder, key)

            # GT: im1
            lq_path = [
                osp.join(key_folder, f'{self.filename_tmpl.format("im1")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im1")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format(im1_right)}')
            ]
            gt_path = osp.join(self.gt_folder, key, 'im1.png')
            data_infos.append(
                dict(
                    lq_path=lq_path,
                    gt_path=gt_path,
                    key=key + '/im1',
                ))

            # GT: im2
            lq_path = [
                osp.join(key_folder, f'{self.filename_tmpl.format("im1")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im2")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im3")}')
            ]
            gt_path = osp.join(self.gt_folder, key, 'im2.png')
            data_infos.append(
                dict(
                    lq_path=lq_path,
                    gt_path=gt_path,
                    key=key + '/im2',
                ))

            # GT: im3
            lq_path = [
                osp.join(key_folder, f'{self.filename_tmpl.format(im3_left)}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im3")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im3")}')
            ]
            gt_path = osp.join(self.gt_folder, key, 'im3.png')
            data_infos.append(
                dict(
                    lq_path=lq_path,
                    gt_path=gt_path,
                    key=key + '/im3',
                ))

        return data_infos


@DATASETS.register_module()
class Vimeo90KTripletSequenceDataset(Vimeo90KTripletCenterGTDataset):
    """
    Difference to Vimeo90KTripletCenterGTDataset:
        1. Load the whole sequence for LQ and GT.
    """

    def load_annotations(self):
        """
        Difference to that of Vimeo90KTripletCenterGTDataset:
            1. Record the whole sequence for GT.
            2. Record only one sequence.
        """
        # get keys
        with open(self.ann_file, 'r') as f:
            keys = f.read().split('\n')
            keys = [
                k.strip() for k in keys if (k.strip() is not None and k != '')
            ]

        data_infos = []
        for key in keys:
            key = key.replace('/', os.sep)
            key_folder = osp.join(self.lq_folder, key)
            gt_folder = osp.join(self.gt_folder, key)

            lq_path = [
                osp.join(key_folder, f'{self.filename_tmpl.format("im1")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im2")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im3")}')
            ]
            gt_path = [
                osp.join(gt_folder, 'im1.png'),
                osp.join(gt_folder, 'im2.png'),
                osp.join(gt_folder, 'im3.png')
            ]
            data_infos.append(dict(
                lq_path=lq_path,
                gt_path=gt_path,
                key=key,
            ))
        return data_infos
