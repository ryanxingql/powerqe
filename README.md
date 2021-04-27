# PowerQE: An Open Framework for Quality Enhancement of Compressed Visual Data

- [PowerQE: An Open Framework for Quality Enhancement of Compressed Visual Data](#powerqe-an-open-framework-for-quality-enhancement-of-compressed-visual-data)
  - [1. Dependencies](#1-dependencies)
    - [Basis](#basis)
    - [Deformable Convolutions V2 (Optional)](#deformable-convolutions-v2-optional)
  - [2. Image Data](#2-image-data)
    - [Download & Unzip & Symbolic-link](#download--unzip--symbolic-link)
    - [Down-sample](#down-sample)
    - [Compress](#compress)
    - [Collect images of different distortions for blind tasks](#collect-images-of-different-distortions-for-blind-tasks)
  - [3. Video Data](#3-video-data)
  - [4. Train](#4-train)
    - [General](#general)
    - [Resume training](#resume-training)
  - [5. Test](#5-test)
  - [6. Results](#6-results)
    - [Baseline](#baseline)
    - [Non-blind image quality enhancement](#non-blind-image-quality-enhancement)
    - [Blind image quality enhancement](#blind-image-quality-enhancement)
    - [Non-blind image perceptual quality enhancement](#non-blind-image-perceptual-quality-enhancement)
    - [Non-blind video quality enhancement](#non-blind-video-quality-enhancement)
  - [7. FAQ](#7-faq)
  - [8. License](#8-license)

:running: An **unified** framework for training/testing blind/non-blind fidelity/perception-oriented **quality enhancement** approaches for compressed images/videos based on PyTorch.

:hammer_and_wrench: Support now:

- [x] [AR-CNN](https://openaccess.thecvf.com/content_iccv_2015/html/Dong_Compression_Artifacts_Reduction_ICCV_2015_paper.html)
- [x] [DCAD](https://ieeexplore.ieee.org/abstract/document/7923714/)
- [x] [DnCNN](https://arxiv.org/abs/1608.03981)
- [x] [CBDNet](https://arxiv.org/abs/1807.04686)
- [x] [RBQE](https://github.com/RyanXingQL/RBQE)
- [x] [ESRGAN](https://github.com/RyanXingQL/SubjectiveQE-ESRGAN)
- [x] S2G: to be released soon.
- [ ] [MFQE](https://github.com/RyanXingQL/MFQEv2.0)
- [ ] [STDF](https://github.com/RyanXingQL/STDF-PyTorch)

:e-mail: Feel free to contact: `ryanxingql@gmail.com`.

## 1. Dependencies

### Basis

```bash
git clone git@github.com:RyanXingQL/PowerQE.git --depth=1 --recursive
cd <path-to-PowerQE>

conda create -n pqe python=3.7 -y
conda activate pqe

# given CUDA 10.1
python -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install tqdm lmdb pyyaml opencv-python scikit-image tensorboard lpips
```

### Deformable Convolutions V2 (Optional)

DCNv2 is required for some methods, e.g., S2G and STDF.

```bash
cd <path-to-PowerQE>/net/ops/dcn
sh build.sh

# check (optional)
python simple_check.py
```

## 2. Image Data

We take the DIV2K dataset for an example.

### Download & Unzip & Symbolic-link

```bash
cd <any-path-to-your-database>
mkdir div2k
cd div2k/

# 800 2K images for training and validation (3.3 GB)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip

# 100 2K images for test (428 MB)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

mkdir raw/
unzip -j DIV2K_train_HR.zip -d raw/
unzip -j DIV2K_valid_HR.zip -d raw/

cd <path-to-PowerQE>
mkdir data
ln -s <path-to-div2k> data/
```

We take `{0001-0700}.png` for training, `{0701-0800}.png` for validation, and `{0801-0900}.png` for test. Note that for other data sets, shuffling may be needed.

### Down-sample

To save GPU memory while keeping the completeness of image semantics, we down-sample these 2K images by the factor of 2.

```bash
cd <path-to-PowerQE>/data_pre_process/
python main_down_sample.py div2k 2
```

### Compress

Then compress these images. We take JPEG (QF=10) and BPG (HEVC-MSP, QP=42) as examples.

```bash
cd <path-to-PowerQE>/data_pre_process/

# JPEG QF=10
python main_compress_jpeg.py div2k/raw_ds_2 10 div2k/jpeg_ds_2

# BPG (HEVC-MSP) QP=42
# given libbpg-0.9.8/ being <dir-to-libbpg>
# check FAQ for details of installing BPG
python main_compress_bpg.py div2k/raw_ds_2 <dir-to-libbpg> 42 div2k/bpg_ds_2
```

### Collect images of different distortions for blind tasks

For blind QE tasks, we collect images of different QP/QFs into new dirs by symbolic links.

```bash
cd <path-to-PowerQE>/data_pre_process/

python main_combine_im.py jpeg

python main_combine_im.py hevc
```

## 3. Video Data

To obtain uncompressed raw videos, we adopt the [MFQEv2 dataset](https://gist.github.com/RyanXingQL/db0b67abb771f02ad9d6c6536eec971e).

Download, unzip and compress the training and test videos following the instructions. It may take days to compress videos due to the low speed of the HM codec.

## 4. Train

### General

Edit YML in `opts/`, especially `bs_pg`, `nworker_pg` and `real_bs_pg`.

`nworker_pg` is recommended to be equal to `bs_pg`.

Then run:

```bash
cd <path-to-PowerQE>
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=11111 train.py --opt opts/arcnn.yml --case div2k-qf10

# check tensorboard
cd <path-to-PowerQE>/exp/arcnn-div2k-qf10
tensorboard --logdir=./
```

You can also use multiple gpus:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=11111 train.py --opt opts/arcnn.yml --case div2k-qf10
```

In general, `real_bs_pg` should be equal to `bs_pg`. If you want to use one GPU to act as N GPUs, just edit the `real_bs_pg` to be N * `bs_pg`.

### Resume training

If you want to resume training from `load_state/ckp_load_path`, edit `load_state/if_load` to be `True`.

## 5. Test

Edit YML in `opts/`, then run:

```bash
cd <path-to-PowerQE>
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=21111 test.py --opt opts/arcnn.yml --case div2k-qf10
```

Pre-trained models: see GitHub releases or [Baidu Cloud](https://pan.baidu.com/s/1aXMn2mi_SWR_W0y2oAL-Fg) (key: prqe).

## 6. Results

Note:

- The unit of PSNR is dB.
- FPS is test with one Tesla V100-SXM2-16GB and Intel (R) Xeon (R) Platinum 8163 CPUs @ 2.50 GHz. FPS can vary depending on many conditions. Just FYI.
- For evaluating perception-driven methods, we adopt two versions of NIQE. The first is NIQE-M, which is built in MATLAB. The other is built in the PIRM 18' challenge. Please refer to [this repository](https://github.com/RyanXingQL/Image-Quality-Assessment-Toolbox) for details of image quality assessment.

### Baseline

<details>
<summary><b>JPEG</b></summary>
<p>

|QF|10|20|30|40|50|
|-:|:-:|:-:|:-:|:-:|:-:|
|PSNR|25.83|28.00|29.18|29.98|30.64|

|metric|SSIM|LPIPS|FID|NIQE-M|PI|NIQE|MA|
|-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|QF=10|0.826|0.292|62.206|5.662|4.863|6.033|6.232|

</p>
</details>

<details>
<summary><b>HEVC</b></summary>
<p>

|QP|22|27|32|37|42|
|-:|:-:|:-:|:-:|:-:|:-:|
|PSNR|38.62|36.31|33.53|30.80|28.35|

|metric|SSIM|LPIPS|FID|NIQE-M|PI|NIQE|MA|
|-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|QP=42|0.890|0.204|41.886|3.697|3.986|4.308|6.257|

</p>
</details>

### Non-blind image quality enhancement

|method|params|FPS|R11|R25|
|-:|:-:|:-:|:-:|:-:|
|**AR-CNN**|20,099|351.8|27.00|28.80|
|**DnCNN***|40,515|136.9|27.28|28.93|
|**DCAD**|40,451|269.4|27.26|28.93|

- R11: PSNR (dB) of enhanced JPEG QF=10 images.
- R25: PSNR (dB) of enhanced HEVC QP=42 images.
- DnCNN* is a modified version of DnCNN, which turns off the batch normalization (BN). See FAQ.

### Blind image quality enhancement

**CBDNet***

|params|FPS|R11|R12|R13|R14|R15|R21|R22|R23|R24|R25|
|-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|1,860,806|120.9|27.70|29.90|31.09|31.92|32.58|40.22|37.69|34.77|31.89|29.21|

- CBDNet* is a modified version of CBDNet, which is trained in an end-to-end manner without total variation (TV) loss on the noise estimation map. See FAQ.
- R11-15: PSNR (dB) of enhanced JPEG QF=10/20/30/40/50 images.
- R21-25: PSNR (dB) of enhanced HEVC QP=22/27/32/37/42 images.

</p>
</details>

**RBQE**

Take a 5-level RBQE network as an example.

|metric|R11|R12|R13|R14|R15|R21|R22|R23|R24|R25|
|-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|PSNR|27.78|30.00|31.24|32.07|32.74|40.15|37.69|34.77|31.90|29.24|
|PSNR-non-blind|27.89|30.05|31.23|31.96|32.43|40.04|37.68|34.84|32.01|29.35|
|FPS*|28.3|28.6|28.3|30.0|24.3|50.02|23.8|27.6|23.6|24.2|

|exit|4|3|2|1|0|
|-:|:-:|:-:|:-:|:-:|:-:|
|params|3,553,430|2,333,996|1,409,672|743,594|298,898|

- PSNR-non-blind: images with QF=50/40/30/20/10 or QP=22/27/32/37/42 exit at exit 1/2/3/4/5 without IQA module.
- FPS*: In the official repo, the MATLAB-based IQAM is fed with Y channel input. Here for simplicity, we re-implement IQAM with Python and feed it with R channel input. Hyper-parameters such as exit thresholds have been tuned over the new input dataset. Since the re-implemented Python-based IQAM is much, much slower than the official MATLAB-based one, we record the FPS without IQAM as FPS*.

<details>
<summary><b>Trade-off</b></summary>
<p>

Fig: ave. delta PSNR of all JPEG-compressed images vs. exit threshold.

![dpsnr-vs-threshold-jpeg](https://user-images.githubusercontent.com/34084019/116222277-09162c80-a781-11eb-832e-1b4c5aa52b33.png)

Fig: ave. delta PSNR of all HEVC-compressed images vs. exit threshold.

![dpsnr-vs-threshold-hevc](https://user-images.githubusercontent.com/34084019/116222260-03b8e200-a781-11eb-8b42-c2aeaacf81a1.png)

</p>
</details>

### Non-blind image perceptual quality enhancement

JPEG (QF=10)

|method|params|FPS|PSNR|SSIM|LPIPS|FID|NIQE-M|PI|NIQE|MA|
|-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|**ESRGAN***|1,451,011|51.4|25.624|0.822|0.147|39.773|3.527|3.226|3.658|7.127|
|**S2G**|18,168,546|4.9|26.048|0.835|0.139|32.601|3.430|3.117|3.500|7.140|

HEVC (QP=42)

|method|PSNR|SSIM|LPIPS|FID|NIQE-M|PI|NIQE|MA|
|-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|**ESRGAN***|27.172|0.865|0.121|32.475|3.423|3.146|3.502|7.112|
|**S2G**|27.466|0.870|0.115|29.945|3.237|3.057|3.385|7.139|

- ESRGAN* is a modified version of ESRGAN for quality enhancement task (ESRGAN is for super-resolution).

### Non-blind video quality enhancement

(To-do)

|method|params|FPS|PSNR (dB, HEVC)|
|-:|:-:|:-:|:-:|
|**MFQEv2**|
|**STDF**|

## 7. FAQ

<details>
<summary><b>How to install BPG?</b></summary>
<p>

Please refer to [BPG official website](https://bellard.org/bpg/) and [BPG specification](https://bellard.org/bpg/bpg_spec.txt) for official instructions.

My solution:

```bash
# install yasm
cd <any-path-you-like>
wget http://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz  
tar -zxvf yasm-1.3.0.tar.gz
cd yasm-1.3.0
./configure
make
sudo make install

# install sdl
sudo apt update
sudo apt-get install libsdl-image1.2-dev

# install libbpg
cd <any-path-you-like>
wget https://bellard.org/bpg/libbpg-0.9.8.tar.gz
tar -zxvf libbpg-0.9.8.tar.gz
cd libbpg-0.9.8/
make
sudo make install
```

</p>
</details>

<details>
<summary><b>Why turn off BN in DnCNN?</b></summary>
<p>

In my view, BN is not suitable for the QE task. The normalization statistics learned from training samples are not in well consistance with those of test samples, due to their different sizes and contents.

I observe from experiments that BN significantly degrades the performances of validation and test for DnCNN.

</p>
</details>

<details>
<summary><b>Why train CBDNet in an end-to-end manner without TV loss?</b></summary>
<p>

In the original paper, the noise estimation sub-net of CBDNet is supervised by the Gaussian deviation map.

However, the compression artifacts in our tasks are not as regular/simple as Gaussian noises. For example, the compression artifacts are mingled with the image content. Thus, it is not suitable to learn a QP/QF map with constant QP/QF value.

An alternative option is to supervise the sub-net by error maps (raw - cmp). However, the whole CBDNet is also designed to learn the error/residual between compressed and raw images. Therefore, the error estimation is indeed a deep supervision that may interfere the feature abstraction process. Thus, it is meaningless and even harmful to do so.

The most important reason is that: the learned noise estimation map may not be that helpful to an end-to-end learning framework. One similar example is the flow estimation in video processing network. As shown in [TOFlow](http://toflow.csail.mit.edu/), the end-to-end optimized flow map is not equal to the real motion map. The end-to-end learned feature map can help tackle the occlusion and other problems. Thus, we train the CBDNet in an end-to-end manner (i.e., remove the TV loss between the estimated map and a constant map).

</p>
</details>

## 8. License

You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing this repository** and **indicating any changes** that you've made.

If you find this repository helpful, you may cite:

```tex
@misc{PowerQE_xing_2021,
  author = {Qunliang Xing},
  title = {PowerQE: An Open Framework for Quality Enhancement of Compressed Visual Data},
  howpublished = "\url{https://github.com/RyanXingQL/PowerQE}",
  year = {2021}, 
  note = "[Online; accessed 11-April-2021]"
}
```
