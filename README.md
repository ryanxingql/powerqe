# PowerQE

This repository supports some quality enhancement approaches for compressed images/videos based on PyTorch and MMEditing (v1.x).

- [ ] [BasicVSR++ (CVPR 2022)](https://arxiv.org/abs/2104.13371)
- [ ] [STDF (AAAI 2020)](https://github.com/ryanxingql/stdf-pytorch)
- [ ] [RBQE (ECCV 2020)](https://arxiv.org/abs/2006.16581)
- [ ] [MFQEv2 (TPAMI 2019)](https://github.com/ryanxingql/mfqev2.0)
- [x] [CBDNet (CVPR 2019)](https://arxiv.org/abs/1807.04686)
- [ ] [EDVR (CVPRW 2019)](https://arxiv.org/abs/1905.02716)
- [x] [ESRGAN (ECCVW 2018)](https://arxiv.org/abs/1809.00219)
- [x] [RDN (CVPR 2018)](https://arxiv.org/abs/1802.08797)
- [x] [DnCNN (TIP 2017)](https://arxiv.org/abs/1608.03981)
- [x] [DCAD (DCC 2017)](https://ieeexplore.ieee.org/abstract/document/7923714)
- [x] [UNet (MICCAI 2015)](https://arxiv.org/abs/1505.04597)
- [x] [AR-CNN (ICCV 2015)](https://arxiv.org/abs/1504.06993)

TODO:

- [ ] Support blind RBQE training and testing.
- [ ] Support RBQE with multiprocessing IQA.

Main difference to MMEditing:

- Support downsampling before enhancement and upsampling after enhancement to save memory.
- Save LQ, GT and output when testing.
- Evaluate "LQ vs. GT" and "output vs. GT" when testing.
- Bug fixed.

MMEditing is a submodule of PowerQE. Users can easily upgrade the MMEditing, and add their own models to PowerQE without modifying the MMEdit package. One should clone PowerQE along with MMEditing like this:

```bash
git clone -b v3-dev --recurse-submodules --depth 1 https://github.com/ryanxingql/powerqe.git
```

## Environment

- `environment.yml`
- `requirements.txt`
- MMEditing (PyTorch + MMCV + MMEdit)

First update your mirror (optional):

- Conda: https://mirrors.tuna.tsinghua.edu.cn/help/anaconda
- pip: https://mirrors.tuna.tsinghua.edu.cn/help/pypi

Then install PowerQE:

```bash
conda env create -f environment.yml  # create the powerqe env
conda activate powerqe
pip install -r requirements.txt
```

Finally install MMEdit following `mmediting/docs/en/install.md`

```bash
conda install pytorch=1.10 torchvision cudatoolkit=11.3 -c pytorch

#pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
# or
pip3 install openmim
mim install mmcv-full==1.5.0

cd mmediting
pip3 install -e .

# verify
cd ~
python -c "import mmedit; print(mmedit.__version__)"
```

## Training

```bash
#chmod +x ./tools/dist_train.sh

conda activate powerqe && \
CUDA_VISIBLE_DEVICES=0 PORT=29500 \
./tools/dist_train.sh \
./configs/rdn/rdn_qe_c64b8_div2k_ps128_bs32_1000k_g1.py \
1
```

Other options:

- `--resume-from <ckp-path>`: Resume training status (model weights, number of iterations, optimizer status, etc.) from a checkpoint file.

## Testing

```bash
#chmod +x ./tools/dist_test.sh

conda activate powerqe && \
CUDA_VISIBLE_DEVICES=0 PORT=29510 \
./tools/dist_test.sh \
./configs/rdn/rdn_qe_c64b8_div2k_ps128_bs32_1000k_g1.py \
./work_dirs/rdn_qe_c64b8_div2k_ps128_bs32_1000k_g1/latest.pth \
1 \
--save-path ./work_dirs/rdn_qe_c64b8_div2k_ps128_bs32_1000k_g1/results/
```

## Q&A

### Crop image border before evaluation

Due to the padding of upsampling, the error at border is significant. PowerQE follows the common practice in SR to crop image border before evaluation.

### Pre-commit hook

PowerQE follows [MMCV](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) to support pre-commit hook. The config file is inherited from [MMEditing](https://github.com/ryanxingql/mmediting/blob/master/.pre-commit-config.yaml). Installation:

```bash
conda activate powerqe
pip install -U pre-commit
pre-commit install
```

On every commit, linters and formatter will be enforced. You can also run hooks manually:

```bash
pre-commit run --all-files
```

### Same items between PowerQE and MMEditing such as `Compose`

When constructing the pipelines for a dataset, the dataset (`BaseDataset` in fact) will refer to `Compose`. Then, `Compose` refers to `..registry` for `PIPELINES`.

PowerQE has its own pipelines such as `PairedCenterCrop`. As a result, PowerQE has to define a new `Compose`, which refers to its own `..registry`.

Note that `Compose` in PowerQE will not be registered into `PIPELINES`.
