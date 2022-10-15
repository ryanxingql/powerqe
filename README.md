# PowerQE

## Environment

- MMEditing (PyTorch + MMCV + MMEditing)

My example:

```bash
conda create --name powerqe python=3.8 -y && conda activate powerqe

# install MMEditing following mmediting/docs/en/install.md

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y

#pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install openmim
mim install mmcv-full==1.5.0

cd mmediting
pip3 install -e .  # everytime you update the submodule mmediting, you have to do this again

# verify
cd ~
python -c "import mmedit; print(mmedit.__version__)"
```

## Training

```bash
conda activate powerqe

#chmod +x ./tools/dist_train.sh

CUDA_VISIBLE_DEVICES=0 PORT=29500 \
./tools/dist_train.sh \
./configs/rdn/rdn_qe_r4c64b8_g1_1000k_div2k.py \
1
```

Other options:

- `--resume-from <ckp-path>`: Resume training status (model weights, number of iterations, optimizer status, etc.) from a checkpoint file.

## Testing

```bash
conda activate powerqe

#chmod +x ./tools/dist_test.sh

CUDA_VISIBLE_DEVICES=0 PORT=29510 \
./tools/dist_test.sh \
./configs/rdn/rdn_qe_r4c64b8_g1_1000k_div2k.py \
./work_dirs/rdn_qe_r4c64b8_g1_1000k_div2k/latest.pth \
1 \
--save-path ./work_dirs/rdn_qe_r4c64b8_g1_1000k_div2k/results/
```

## Q&A

### Main difference between PowerQE and MMEditing

- Support downsampling before enhancement and upsampling after enhancement to save memory.
- Save LQ, GT and output when testing.
- Evaluate "LQ vs. GT" and "output vs. GT" when testing.

### Crop image border before evaluation

Due to the padding of upsampling, the error at border is significant. We follow the common practice in SR to crop image border before evaluation.
