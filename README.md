# PowerQE

This repository supports some quality enhancement approaches for compressed images/videos based on PyTorch and MMEditing (v0.x).

- [ ] [BasicVSR++ @ CVPR 22'](https://arxiv.org/abs/2104.13371)
- [x] [MPRNet @ CVPR 21'](https://github.com/swz30/MPRNet)
- [ ] [STDF @ AAAI 20](https://github.com/ryanxingql/stdf-pytorch)
- [x] [RBQE @ ECCV 20'](https://arxiv.org/abs/2006.16581): non-blind version.
- [ ] [MFQEv2 @ TPAMI 19'](https://github.com/ryanxingql/mfqev2.0)
- [x] [CBDNet @ CVPR 19'](https://arxiv.org/abs/1807.04686): non-blind version.
- [ ] [EDVR @ CVPR 19'](https://arxiv.org/abs/1905.02716)
- [x] [SAN @ CVPR 19'](https://github.com/daitao/SAN)
- [x] [ESRGAN @ ECCVW 18'](https://arxiv.org/abs/1809.00219)
- [x] [RDN @ CVPR 18'](https://arxiv.org/abs/1802.08797)
- [x] [DnCNN @ TIP 17'](https://arxiv.org/abs/1608.03981)
- [x] [DCAD @ DCC 17'](https://ieeexplore.ieee.org/abstract/document/7923714)
- [x] [UNet @ MICCAI 15'](https://arxiv.org/abs/1505.04597)
- [x] [AR-CNN @ ICCV 15'](https://arxiv.org/abs/1504.06993)

MMEditing is a submodule of PowerQE. One can easily upgrade the MMEditing, and add their models to PowerQE without modifying the MMEditing repository. One should clone PowerQE along with MMEditing like this:

```bash
git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/ryanxingql/powerqe.git
```

## Environment

- environment.yml
- requirements.txt
- MMEditing (PyTorch + MMCV + MMEdit)

First, update mirrors (optional):

- Conda: https://mirrors.tuna.tsinghua.edu.cn/help/anaconda
- pip: https://mirrors.tuna.tsinghua.edu.cn/help/pypi

Then, create a PowerQE environment:

```bash
conda env create -f environment.yml  # create the powerqe env
conda activate powerqe
```

Next, install MMEditing following mmediting/docs/en/install.md.

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
# or
#pip3 install openmim
#mim install mmcv-full==1.7.0

cd mmediting/
pip3 install -e .

# verify
cd ~
python -c "import mmedit; print(mmedit.__version__)"
```

Finally,

```
cd powerqe/
pip3 install -r requirements.txt
```

## Data

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

## Training

```bash
#chmod +x ./tools/dist_train.sh

conda activate powerqe && \
CUDA_VISIBLE_DEVICES=0 \
PORT=29500 \
./tools/dist_train.sh \  # main script
./configs/<config>.py \  # config path
1 \  # use one gpu
<other-options>  # optional
```

Other options:

- `--resume-from <ckp>.pth`: To resume the training status (model weights, number of iterations, optimizer status, etc.) from a checkpoint file.

## Testing

```bash
#chmod +x ./tools/dist_test.sh

conda activate powerqe && \
CUDA_VISIBLE_DEVICES=0 \
PORT=29510 \
./tools/dist_test.sh \  # main script
./configs/<config>.py \  # config path
./work_dirs/<ckp>.pth \  # checkpoint path
1 \  # use one gpu
<other-options>  # optional
```

Other options:

- `--save-path <save-folder>`: To save output images.

## Q&A

See [Wiki](./docs/v3.md).
