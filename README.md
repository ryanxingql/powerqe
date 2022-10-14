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
CUDA_VISIBLE_DEVICES=0 PORT=29600 ./tools/dist_train.sh ./configs/rdn/rdn_qe_r4c64b8_g1_1000k_div2k.py 1
```
