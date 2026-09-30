
  
# DGenMotion

## KIT FSQ / JiT / APG experiments (2026-09-30)

The current experiment implementation is documented in
[experiments/kit_epoch_20260929/README.md](experiments/kit_epoch_20260929/README.md).
It trains two generators from scratch with a shared frozen KIT FSQ autoencoder,
evaluates after every epoch, and evaluates direct APG on the same JiT EMA weights.
Latest recovery checkpoints and independent CFG/APG best checkpoints are retained.
FSQ, JiT-style clean prediction, and APG are adaptations of existing methods;
the code does not establish that this combination is an original algorithm.

[KIT progress snapshot](doc/KIT_PROGRESS_20260930.md) records incomplete validation
results, not final test results. The APG benefit observed on HumanML must not be
assumed to transfer to KIT. The registered Joint denoiser is a separate historical
experiment and is not used by these two KIT training configurations.

The instructions below are historical setup documentation. `environment.yml`
is a recovered Linux environment export, not a verified clean-install lock for
the current experiment. Follow the experiment README for its additional assets,
official OpenAI CLIP dependency, checks, and launch commands. Preserve existing
datasets and checkpoints when preparing an environment.

### 1. Conda Environment
```bash
conda env create -f environment.yml
conda activate MARDM
```
We test our code on Python 3.10.13, PyTorch 2.2.0, and CUDA 12.1

### 2. Models and Dependencies

#### Download Evaluation Models
```bash
mkdir -p checkpoints
cd checkpoints
mkdir t2m
mkdir kit

cd t2m 
echo -e "Downloading evaluation models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1ejiz4NvyuoTj3BIdfNrTFFZBZ-zq4oKD/view?usp=sharing
echo -e "Unzipping humanml3d evaluators"
unzip evaluators_humanml3d.zip

echo -e "Cleaning humanml3d evaluators zip"
rm evaluators_humanml3d.zip

cd ../kit/
echo -e "Downloading pretrained models for KIT-ML dataset"
gdown --fuzzy https://drive.google.com/file/d/1kobWYZdWRyfTfBj5YR_XYopg9YZLdfYh/view?usp=sharing

echo -e "Unzipping kit evaluators"
unzip evaluators_kit.zip

echo -e "Cleaning kit evaluators zip"
rm evaluators_kit.zip

cd ../../
```

#### Download GloVe
```bash
# Preserve any existing glove directory.
echo -e "Downloading glove (in use only by the evaluators)"
gdown --fuzzy https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing

unzip glove.zip
echo -e "Cleaning GloVe zip\n"
rm glove.zip

echo -e "Downloading done!"
```

#### Download Pre-trained Models
```bash
cd checkpoints/t2m
echo -e "Downloading pretrained models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1TBybFByAd-kD4AuFgMyR3ZBt4VV43Sif/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1csjlxi0uOhfPPEwiThsR0gaj7_VDmgb6/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1nWoEcN4rEFKi4Xyf_ObKinDmSQNPKXgU/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1nfX_j8VzMmynqKv8x68pXrsL3c0qWLXA/view?usp=sharing
echo -e "Unzipping"
unzip MARDM_SiT_XL.zip
unzip MARDM_DDPM_XL.zip
unzip length_estimator.zip
unzip AE_humanml3d.zip
echo -e "Cleaning zips"
rm MARDM_SiT_XL.zip
rm MARDM_DDPM_XL.zip
rm length_estimator.zip
rm AE_humanml3d.zip

cd ../../
```

### 3. Obtain Data
**You do not need to get data** if you only want to generate motions based on textual instructions.

If you want to reproduce and evaluate our method, you can obtain both 
**HumanML3D** and **KIT** following instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git). By default, the data path is set to `./datasets`.

For dataset Mean and Std, you are welcome to use the eval_mean,npy and eval_std,npy in the utils,
or you can calculate based on your obtained dataset using:
```
python utils/cal_mean_std.py
```
</details>

## 💻  Demo
<details>

### (a) Generate with single textual instruction
```bash
python sample.py --name MARDM_SiT_XL --text_prompt "A person is running on a treadmill."
```
### (b) Generate from a prompt file
in a txt file, in each line, your input should be `<text description>#<motion length>`,
you can push NA as motion length to let model determine the motion length
(if there is **one** NA in file, all the others will be **NA** as well).

```bash
python sample.py --name MARDM_SiT_XL --text_path ./text_prompt.txt
```
</details>

## 🎆 Train Your Own MARDM models
<details>

### HumanML3D
#### AE
```bash
python train_AE.py --name AE --dataset_name t2m --batch_size 256 --epoch 50 --lr_decay 0.05
```
#### MARDM
```bash
# MARDM SiT-based (best results)
python train_MARDM.py --name MARDM_SiT_XL --model "MARDM-SiT-XL" --dataset_name t2m --batch_size 64 --ae_name AE
# MARDM DDPM-based
python train_MARDM.py --name MARDM_DDPM_XL --model "MARDM-DDPM-XL" --dataset_name t2m --batch_size 64 --ae_name AE
```

### KIT-ML
#### AE
```bash
python train_AE.py --name AE --dataset_name kit --batch_size 512 --epoch 50 --lr_decay 0.1
```
#### MARDM
```bash
# MARDM SiT-based (best results)
python train_MARDM.py --name MARDM_SiT_XL --model "MARDM-SiT-XL" --dataset_name kit --batch_size 16 --ae_name AE --milestones 20000
# MARDM DDPM-based
python train_MARDM.py --name MARDM_DDPM_XL --model "MARDM-DDPM-XL" --dataset_name kit --batch_size 16 --ae_name AE --milestones 20000
```
</details>

## 📖 Evaluate MARDM models
<details>

### HumanML3D
#### AE
```bash
python evaluation_AE.py --name AE --dataset_name t2m
```
#### MARDM
```bash
# MARDM SiT-based (best results)
python evaluation_MARDM.py --name MARDM_SiT_XL --model "MARDM-SiT-XL" --dataset_name t2m --cfg 4.5
# MARDM DDPM-based
python evaluation_MARDM.py --name MARDM_DDPM_XL --model "MARDM-DDPM-XL" --dataset_name t2m --cfg 4.5
```
### KIT-ML
#### AE
```bash
python evaluation_AE.py --name AE --dataset_name kit
```
#### MARDM
```bash
# MARDM SiT-based (best results)
python evaluation_MARDM.py --name MARDM_SiT_XL --model "MARDM-SiT-XL" --dataset_name kit --cfg 2.5
# MARDM DDPM-based
python evaluation_MARDM.py --name MARDM_DDPM_XL --model "MARDM-DDPM-XL" --dataset_name kit --cfg 2.5
```
</details>

## 🎏 Temporal Editing
<details>

```bash
python edit.py --name MARDM_SiT_XL -msec 0.3,0.6 --text_prompt "A man dancing around." --source_motion 000612.npy
```
</details>

