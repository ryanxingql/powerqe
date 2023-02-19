# PowerQE

This repository supports some quality enhancement approaches for compressed images/videos based on PyTorch and MMEditing (v0.x).

- [ ] [BasicVSR++ (CVPR 2022)](https://arxiv.org/abs/2104.13371)
- [x] [MPRNet (CVPR 2021)](https://github.com/swz30/MPRNet)
- [ ] [STDF (AAAI 2020)](https://github.com/ryanxingql/stdf-pytorch)
- [ ] [RBQE (ECCV 2020)](https://arxiv.org/abs/2006.16581)
- [ ] [MFQEv2 (TPAMI 2019)](https://github.com/ryanxingql/mfqev2.0)
- [x] [CBDNet (CVPR 2019)](https://arxiv.org/abs/1807.04686)
- [ ] [EDVR (CVPRW 2019)](https://arxiv.org/abs/1905.02716)
- [x] [SAN (CVPR 2019)](https://github.com/daitao/SAN)
- [x] [ESRGAN (ECCVW 2018)](https://arxiv.org/abs/1809.00219)
- [x] [RDN (CVPR 2018)](https://arxiv.org/abs/1802.08797)
- [x] [DnCNN (TIP 2017)](https://arxiv.org/abs/1608.03981)
- [x] [DCAD (DCC 2017)](https://ieeexplore.ieee.org/abstract/document/7923714)
- [x] [UNet (MICCAI 2015)](https://arxiv.org/abs/1505.04597)
- [x] [AR-CNN (ICCV 2015)](https://arxiv.org/abs/1504.06993)

The main difference to MMEditing:

- Support down-sampling before enhancement and up-sampling after enhancement to save memory.
- Save LQ, GT and output when testing.
- Evaluate "LQ vs. GT" and "output vs. GT" when testing.
- Bug fixed.

MMEditing is a submodule of PowerQE. One can easily upgrade the MMEditing, and add their models to PowerQE without modifying the MMEditing repository. One should clone PowerQE along with MMEditing like this:

```bash
git clone -b v3 --recurse-submodules --depth 1 https://github.com/ryanxingql/powerqe.git
```

## Environment

- `environment.yml`
- `requirements.txt`
- MMEditing (PyTorch + MMCV + MMEdit)

First, update mirrors (optional):

- Conda: https://mirrors.tuna.tsinghua.edu.cn/help/anaconda
- pip: https://mirrors.tuna.tsinghua.edu.cn/help/pypi

Then, create a PowerQE environment:

```bash
conda env create -f environment.yml  # create the powerqe env
conda activate powerqe
```

Next, install MMEditing following `mmediting/docs/en/install.md`

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

See [WIKI](https://github.com/ryanxingql/powerqe/wiki/V3-documentation).
