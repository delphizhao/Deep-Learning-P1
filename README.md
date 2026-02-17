# Deep-Learning-P1
UAB Team 5

Team Members： Fujian Zhao, Tianwen Wang , Yaqi Wan

Python Version:3.13

H. pylori Patch Classification (Binary)
1. Project Goal

We perform binary classification on histology image patches to predict H. pylori presence.

Label definition (二分类标签定义)

Presence = 1 → Positive (HP present)

Presence = -1 → Negative (HP absent)

Other values are ignored.

We train and compare two CNN models:

ResNet18 (baseline)

ResNet50 (pretrained on ImageNet) (stronger baseline)

2. Dataset & Files
2.1 Provided metadata files

Put these files under data/ (or update paths accordingly):

data/HP_WSI-CoordAnnotatedAllPatches.xlsx
Patch-level annotations including Pat_ID, Section_ID, Window_ID, Presence, etc.

data/PatientDiagnosis.csv (optional)
Patient-level diagnosis. If missing, our code falls back to patch-level split.

2.2 Image root directory (重要：图片路径)

Images live on the GPU server under something like:

/fhome/vlia/HelicoDataSet/CrossValidation/Cropped/

Expected structure (example):

Cropped/
  B22-83_1/
    0.png
    1.png
    ...
  B22-84_1/
    ...


Important: The training pipeline will only use images that exist under --data_root.
If the path mapping is wrong, you may end up training on only a few hundred images.

3. Recommended Repo Layout

Suggested structure:

Deep-Learning-P1/
  src/
    prepare_splits.py
    dataset_csv.py
    train_cnn.py
    metrics.py
    ...
  data/
    HP_WSI-CoordAnnotatedAllPatches.xlsx
    PatientDiagnosis.csv   (optional)
  splits/                  (generated)
  runs/                    (generated)

4. Environment Setup (GPU server)
4.1 Python

On Ubuntu server, use python3 (NOT python):

python3 --version


Install deps:

pip3 install -r requirements.txt


If no requirements file, at least:

pip3 install torch torchvision pandas openpyxl pillow

4.2 GPU check

Select a GPU and verify CUDA:

export CUDA_VISIBLE_DEVICES=1

python3 - << 'EOF'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
EOF

5. Step 1 — Generate Train/Val/Test Splits

Run:

python3 src/prepare_splits.py \
  --data_root /fhome/vlia/HelicoDataSet/CrossValidation/Cropped \
  --xlsx data/HP_WSI-CoordAnnotatedAllPatches.xlsx \
  --patient_csv data/PatientDiagnosis.csv \
  --out_dir splits


Output:

splits/train.csv

splits/val.csv

splits/test.csv

Sanity check:

head -n 3 splits/train.csv
wc -l splits/train.csv splits/val.csv splits/test.csv


Note:
If PatientDiagnosis.csv is missing, the script may warn and fall back to patch-level splitting.

6. Step 2 — Train Model #1 (ResNet18 baseline)
export CUDA_VISIBLE_DEVICES=1

python3 src/train_cnn.py \
  --train_csv splits/train.csv \
  --val_csv splits/val.csv \
  --test_csv splits/test.csv \
  --model resnet18 \
  --epochs 20 \
  --batch_size 16 \
  --lr 1e-4 \
  --num_workers 4 \
  --out_dir runs


Outputs (example):

runs/resnet18_seed42/best.pth

console logs: train loss/acc + val loss/acc + final test metrics

7. Step 3 — Train Model #2 (ResNet50 pretrained)
export CUDA_VISIBLE_DEVICES=1

python3 src/train_cnn.py \
  --train_csv splits/train.csv \
  --val_csv splits/val.csv \
  --test_csv splits/test.csv \
  --model resnet50 \
  --pretrained \
  --epochs 20 \
  --batch_size 16 \
  --lr 1e-4 \
  --num_workers 4 \
  --out_dir runs


This downloads ImageNet weights automatically (first run only).

8. Metrics & Reporting
8.1 What to report (PPT/Report)

For each model, report:

Train/Val accuracy curves (or best val accuracy)

Final test accuracy

Confusion matrix (optional but recommended)

Dataset size used (IMPORTANT: how many images actually used)

8.2 Why pretrained still needs training

Pretraining provides generic visual features (edges/textures).
We still need fine-tuning to adapt to histology staining patterns and our binary label.

9. Common Issues & Fixes
9.1 “I thought we have tens of thousands, but training uses only hundreds”

This happens when:

--data_root points to wrong folder

path rule {Pat_ID}_{Section_ID}/{Window_ID}.png mismatches real filenames

many images are missing/not downloaded locally

Fix:

ensure data_root contains the full dataset (server path)

verify actual count:

find /fhome/vlia/HelicoDataSet/CrossValidation/Cropped -name "*.png" | wc -l


ensure splits/*.csv contains valid absolute path entries

9.2 “python: command not found”

Use python3 on Ubuntu.

9.3 “Window_ID is not numeric (e.g., 902_Aug1)”

Some metadata fields may be strings. Path generation must not force int() if not guaranteed numeric.

10. Reproducibility (Seeds)

Use consistent seeds in training for fair comparison.
We store checkpoints under runs/<model>_seed<seed>/.

11. Quick Commands Summary

Generate splits:

python3 src/prepare_splits.py --data_root ... --xlsx ... --out_dir splits


Train baseline:

python3 src/train_cnn.py --model resnet18 ...


Train pretrained:

python3 src/train_cnn.py --model resnet50 --pretrained ...
