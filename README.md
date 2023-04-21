# PowerQE

This repository supports some quality enhancement approaches for compressed image/video based on PyTorch and MMEditing.

- [ ] [ProVQE @ CVPRW'22](https://github.com/ryanxingql/winner-ntire22-vqe)
- [ ] [BasicVSR++ @ CVPR'22](https://arxiv.org/abs/2104.13371)
- [x] [MPRNet @ CVPR'21](https://github.com/swz30/MPRNet)
- [ ] [STDF @ AAAI'20](https://github.com/ryanxingql/stdf-pytorch)
- [x] [RBQE @ ECCV'20](https://arxiv.org/abs/2006.16581): non-blind version.
- [ ] [MFQEv2 @ TPAMI'19](https://github.com/ryanxingql/mfqev2.0)
- [x] [CBDNet @ CVPR'19](https://arxiv.org/abs/1807.04686): non-blind version.
- [ ] [EDVR @ CVPR'19](https://arxiv.org/abs/1905.02716)
- [x] [SAN @ CVPR'19](https://github.com/daitao/SAN)
- [x] [ESRGAN @ ECCVW'18](https://arxiv.org/abs/1809.00219)
- [x] [RDN @ CVPR'18](https://arxiv.org/abs/1802.08797)
- [x] [DnCNN @ TIP'17](https://arxiv.org/abs/1608.03981)
- [x] [DCAD @ DCC'17](https://ieeexplore.ieee.org/abstract/document/7923714)
- [x] [U-Net @ MICCAI'15](https://arxiv.org/abs/1505.04597)
- [x] [AR-CNN @ ICCV'15](https://arxiv.org/abs/1504.06993)

## Installation

MMEditing is a submodule of PowerQE. One can easily upgrade the MMEditing, and add their models to PowerQE without modifying the MMEditing repository. One should clone PowerQE along with MMEditing like this:

```bash
git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/ryanxingql/powerqe.git
```

Create environment:

- environment.yml
- MMEditing (PyTorch + MMCV + MMEdit v0.x)
- requirements.txt

Please refer to the [document](./docs/v3.md#env) for detailed installation.

## Prepare data

Create a symbolic link to your data:

```bash
cd powerqe/
ln -s <absolute-path-to-your-data> ./data
```

Place your data like this:

```txt
powerqe/
`-- data/
    `-- div2k/
        |-- train/
        |   |-- gt/
        |   |   |-- 0001.png
        |   |   |-- ...
        |   |   `-- 0800.png
        |   `-- lq/
        |       |-- 0001.png
        |       |-- ...
        |       `-- 0800.png
        `-- valid/
            |-- gt/
            |   |-- 0801.png
            |   |-- ...
            |   `-- 0900.png
            `-- lq/
                |-- 0801.png
                |-- ...
                `-- 0900.png
```

Please refer to the [document](./docs/v3.md#compress) for details about image/video compression.

## Training

```bash
#chmod +x ./tools/dist_train.sh  # for the first time

conda activate powerqe && \
CUDA_VISIBLE_DEVICES=0 \  # use GPU 0
PORT=29500 \  # use port 29500 for communication
./tools/dist_train.sh \  # training script
./configs/<config>.py \  # config path
1 \  # use one gpu
<optional-options>
```

Optional options:

- `--resume-from <ckp>.pth`: To resume the training status (model weights, number of iterations, optimizer status, etc.) from a checkpoint file.

## Testing

```bash
#chmod +x ./tools/dist_test.sh  # for the first time

conda activate powerqe && \
CUDA_VISIBLE_DEVICES=0 \  # use GPU 0
PORT=29510 \  # use port 29510 for communication
./tools/dist_test.sh \  # test script
./configs/<config>.py \  # config path
./work_dirs/<ckp>.pth \  # checkpoint path
1 \  # use one gpu
<optional-options>
```

Optional options:

- `--save-path <save-folder>`: To save output images.
