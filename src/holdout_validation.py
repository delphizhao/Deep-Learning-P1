import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import glob
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# 1. 核心路径配置
# ==========================================
MODEL_PATH = "/import2/hhome/ricse05/Deep-Learning-P1/runs/resnet18_seed42/best.pth"
IMG_ROOT = "/fhome/vlia/HelicoDataSet"
HOLDOUT_DIR = "HoldOut"
OUTPUT_CSV = "holdout_validation_report.csv"

# HoldOut 内部拆分比例：60% 定阈值，40% 最终测试
THRESHOLD_SPLIT = 0.6
RANDOM_SEED = 42

# ==========================================
# 2. 图像预处理
# ==========================================
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class FastInferenceDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_paths[idx]).convert('RGB')
            return preprocess(img)
        except Exception:
            return torch.zeros(3, 256, 256)

def infer_patient(folder, model, device):
    """对单个病人文件夹跑推理，返回阳性比例"""
    all_imgs = glob.glob(os.path.join(folder, "*.png"))
    if not all_imgs:
        return None, 0

    ds = FastInferenceDataset(all_imgs)
    dl = DataLoader(ds, batch_size=128, num_workers=4, pin_memory=True)

    pos_count = 0
    with torch.no_grad():
        for batch_imgs in dl:
            batch_imgs = batch_imgs.to(device)
            outputs = model(batch_imgs)
            preds = torch.max(outputs, 1)[1]
            pos_count += torch.sum(preds == 1).item()

    return len(all_imgs), pos_count / len(all_imgs)

def find_best_threshold(df):
    """在阈值定标集上寻找最优阈值"""
    best_acc = 0
    best_t = 0
    thresholds = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    print(f"\n{'阈值':<10} | {'准确率':<10} | {'敏感度':<10} | {'特异度':<10} | {'F1'}")
    print("-" * 58)

    for t in thresholds:
        df['Pred'] = (df['Positive_Ratio'] > t).astype(int)
        acc  = (df['Pred'] == df['GT_Label']).mean()

        pos_df = df[df['GT_Label'] == 1]
        sens = (pos_df['Pred'] == 1).mean() if len(pos_df) > 0 else 0

        neg_df = df[df['GT_Label'] == 0]
        spec = (neg_df['Pred'] == 0).mean() if len(neg_df) > 0 else 0

        tp = ((df['Pred'] == 1) & (df['GT_Label'] == 1)).sum()
        fp = ((df['Pred'] == 1) & (df['GT_Label'] == 0)).sum()
        fn = ((df['Pred'] == 0) & (df['GT_Label'] == 1)).sum()
        f1 = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0

        marker = " ◀" if acc > best_acc else ""
        print(f"{t:10.1%} | {acc:10.2%} | {sens:10.2%} | {spec:10.2%} | {f1:6.2%}{marker}")

        if acc > best_acc:
            best_acc = acc
            best_t = t

    return best_t

def run_holdout_validation():
    # ------------------------------------------
    # 3. 加载模型
    # ------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print(f"📦 模型加载完毕\n")

    # ------------------------------------------
    # 4. 扫描 HoldOut，对所有病人跑推理
    # ------------------------------------------
    holdout_path = os.path.join(IMG_ROOT, HOLDOUT_DIR)
    all_folders = sorted(glob.glob(os.path.join(holdout_path, "*")))
    all_folders = [p for p in all_folders if os.path.isdir(p)]
    print(f"🔍 HoldOut 共 {len(all_folders)} 位病人，全部推理中...\n")

    all_results = []
    for folder in tqdm(all_folders, desc="推理进度"):
        folder_name = os.path.basename(folder)
        suffix = folder_name.rsplit('_', 1)[-1]
        if suffix not in ('0', '1'):
            continue
        gt_label = int(suffix)
        codi = folder_name.rsplit('_', 1)[0]

        total_patches, pos_ratio = infer_patient(folder, model, device)
        if total_patches is None:
            continue

        all_results.append({
            'Pat_ID': codi,
            'Folder': folder_name,
            'GT_Label': gt_label,
            'Total_Patches': total_patches,
            'Positive_Ratio': pos_ratio,
        })

    full_df = pd.DataFrame(all_results)
    print(f"\n✅ 推理完成，共 {len(full_df)} 位病人")

    # ------------------------------------------
    # 5. 按标签分层拆分：60% 定阈值，40% 最终测试
    #    分层保证两个子集的阳性/阴性比例一致
    # ------------------------------------------
    random.seed(RANDOM_SEED)

    pos_patients = full_df[full_df['GT_Label'] == 1]['Pat_ID'].tolist()
    neg_patients = full_df[full_df['GT_Label'] == 0]['Pat_ID'].tolist()
    random.shuffle(pos_patients)
    random.shuffle(neg_patients)

    n_pos_thresh = int(len(pos_patients) * THRESHOLD_SPLIT)
    n_neg_thresh = int(len(neg_patients) * THRESHOLD_SPLIT)

    thresh_ids = set(pos_patients[:n_pos_thresh] + neg_patients[:n_neg_thresh])
    test_ids   = set(pos_patients[n_pos_thresh:] + neg_patients[n_neg_thresh:])

    thresh_df = full_df[full_df['Pat_ID'].isin(thresh_ids)].copy()
    test_df   = full_df[full_df['Pat_ID'].isin(test_ids)].copy()

    print(f"\n📊 数据集拆分结果 (seed={RANDOM_SEED}):")
    print(f"   阈值定标集: {len(thresh_df)} 人  (阳性:{thresh_df['GT_Label'].sum()}  阴性:{(thresh_df['GT_Label']==0).sum()})")
    print(f"   最终测试集: {len(test_df)} 人  (阳性:{test_df['GT_Label'].sum()}  阴性:{(test_df['GT_Label']==0).sum()})")

    # ------------------------------------------
    # 6. 在定标集上找最优阈值
    # ------------------------------------------
    print(f"\n{'='*58}")
    print("🔧 Step 1：在阈值定标集上寻找最优阈值")
    print(f"{'='*58}")
    best_threshold = find_best_threshold(thresh_df)
    print(f"\n🏆 锁定最优阈值: {best_threshold:.1%}")

    # ------------------------------------------
    # 7. 在最终测试集上评估
    # ------------------------------------------
    print(f"\n{'='*58}")
    print("🎯 Step 2：在最终测试集上盲测（阈值已锁定）")
    print(f"{'='*58}")

    test_df['Pred'] = (test_df['Positive_Ratio'] > best_threshold).astype(int)
    test_df['Correct'] = (test_df['Pred'] == test_df['GT_Label']).astype(int)

    total   = len(test_df)
    correct = test_df['Correct'].sum()
    acc     = correct / total if total > 0 else 0

    pos_df = test_df[test_df['GT_Label'] == 1]
    sens   = (pos_df['Pred'] == 1).mean() if len(pos_df) > 0 else 0

    neg_df = test_df[test_df['GT_Label'] == 0]
    spec   = (neg_df['Pred'] == 0).mean() if len(neg_df) > 0 else 0

    tp = ((test_df['Pred'] == 1) & (test_df['GT_Label'] == 1)).sum()
    fp = ((test_df['Pred'] == 1) & (test_df['GT_Label'] == 0)).sum()
    fn = ((test_df['Pred'] == 0) & (test_df['GT_Label'] == 1)).sum()
    f1 = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0

    print(f"\n  使用阈值:            {best_threshold:.1%}")
    print(f"  最终测试病人数:      {total} 人")
    print(f"    - 阳性 (GT=1):     {len(pos_df)} 人")
    print(f"    - 阴性 (GT=0):     {len(neg_df)} 人")
    print(f"  ─────────────────────────────────────")
    print(f"  Patient-Level 准确率:  {acc:.2%}  ({correct}/{total})")
    print(f"  敏感度 (Sensitivity):  {sens:.2%}")
    print(f"  特异度 (Specificity):  {spec:.2%}")
    print(f"  F1 Score:              {f1:.2%}")
    print(f"{'='*58}")

    # 保存完整报告
    full_df['Split'] = full_df['Pat_ID'].apply(
        lambda x: 'threshold_set' if x in thresh_ids else 'test_set'
    )
    full_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 完整报告已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_holdout_validation()