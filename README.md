# PowerQE: An Open Framework for Quality Enhancement of Compressed Images


:rocket: **Update** (22/4/24): We implement video approaches including MFQEv2 and STDF based on MMEditing at [PowerVQE](https://github.com/ryanxingql/powervqe).

:muscle: **PowerQE** is an **unified** framework for training/testing blind/non-blind fidelity/perception-oriented **quality enhancement** approaches for compressed images based on PyTorch.

:hammer_and_wrench: Support now:

- [x] [AR-CNN](https://openaccess.thecvf.com/content_iccv_2015/html/Dong_Compression_Artifacts_Reduction_ICCV_2015_paper.html)
- [x] [DCAD](https://ieeexplore.ieee.org/abstract/document/7923714/)
- [x] [DnCNN](https://arxiv.org/abs/1608.03981)
- [x] [CBDNet](https://arxiv.org/abs/1807.04686)
- [x] [RBQE](https://github.com/ryanxingql/rbqe)
- [x] [ESRGAN](https://github.com/ryanxingql/subjectiveqe-esrgan)

:rocket: Clone: this repository adopts [PythonUtils](https://github.com/ryanxingql/pythonutils) as a sub-module. You may clone PowerQE by:

```bash
git clone git@github.com:ryanxingql/powerqe.git --depth=1 --recursive
```

:notebook: [[Wiki]](https://github.com/ryanxingql/powerqe/wiki)

:e-mail: Feel free to contact: `ryanxingql@gmail.com`.

## 0. Archive

- v2: support full-resolution DIV2K data-set.
- [v1](https://github.com/ryanxingql/powerqe/tree/ea903fd0d04154c95b321b5100540249856bd44b): support down-sampled DIV2K data-set.

## 1. Dependency

```bash
conda create -n pqe python=3.7 -y && conda activate pqe

# case 1: given CUDA 10.x
python -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install tqdm lmdb pyyaml opencv-python scikit-image tensorboard lpips

# case 2: given CUDA 11.x
python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install tqdm lmdb pyyaml opencv-python scikit-image tensorboard lpips matplotlib
```

## 2. Image Data

We take the DIV2K dataset for an example.

### Download & Unzip & Symlink

```bash
cd <any-path-to-your-database>
mkdir div2k && cd div2k/

# 800 2K images for training and validation (3.3 GB)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip

# 100 2K images for test (428 MB)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

mkdir raw/
unzip -j DIV2K_train_HR.zip -d raw/
unzip -j DIV2K_valid_HR.zip -d raw/

cd <path-to-powerqe>
mkdir data && ln -s <path-to-div2k> data/
```

We take `{0001-0700}.png` for training, `{0701-0800}.png` for validation, and `{0801-0900}.png` for test.

### Compress

We take JPEG (QF=10) and BPG (HEVC-MSP, QP=42) as examples.

[[How to install BPG]](https://github.com/ryanxingql/powerqe/wiki/How-to-install-BPG%3F)

```bash
cd ./data_pre_process/ && conda activate pqe

# JPEG QF=10
python main_compress_jpeg.py div2k raw jpeg 10

# BPG (HEVC-MSP) QP=42
python main_compress_bpg.py div2k raw bpg 42 <dir-to-libbpg>
```

### Combine images of different distortions (required only for blind QE)

For blind QE tasks, we combine images of different QP/QFs into new dirs by symlink.

```bash
cd ./data_pre_process/ && conda activate pqe

python main_compress_jpeg_blind.py div2k raw jpeg  # compress images with qf=10, 20, 30, 40 and 50 first
python main_combine_im.py div2k raw jpeg

python main_compress_bpg_blind.py div2k raw bpg <dir-to-libbpg>  # compress images with qp=42, 37, 32, 27 and 22 first
python main_combine_im.py div2k raw bpg
```

## 3. Train

Edit YML in `opts/`, then run:

```bash
conda activate pqe && CUDA_VISIBLE_DEVICES=0 python train.py -opt opts/arcnn.yml -case div2k_qf10
```

You can also use multiple gpus:

```bash
conda activate pqe && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1111 train.py -opt opts/arcnn.yml -case div2k_qf10
```

Tensorboard:

```bash
cd ./exp/arcnn_div2k_qf10
conda activate pqe && tensorboard --logdir=./ --port=10001 --bind_all
```

## 4. Test

Edit YML in `opts/`, then run:

```bash
conda activate pqe && CUDA_VISIBLE_DEVICES=0 python test.py -opt opts/arcnn.yml -case div2k_qf10
```

## 5. Result

- [[Numeric result]](https://github.com/ryanxingql/powerqe/wiki/Result)
- [[Pre-trained model]](https://github.com/ryanxingql/powerqe/releases)
- [[Validation curve in training]](https://github.com/ryanxingql/powerqe/issues/2)

## 6. License

We adopt Apache License v2.0.

If you find this repository helpful, you may cite:

```tex
@software{2021xing,
  author = {Xing, Qunliang},
  month = {4},
  title = {{An unified framework of quality enhancement approaches for compressed images/videos based on PyTorch}},
  url = {https://github.com/ryanxingql/powerqe},
  version = {1.0.0},
  year = {2021}
}
```
