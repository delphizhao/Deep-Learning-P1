import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# 1. 核心路径配置
# ==========================================
MODEL_PATH = "runs/patch_classifier_phase1/resnet18_seed42/best.pth"
IMG_ROOT = "/fhome/vlia/HelicoDataSet"

# HoldOut 直接在根目录下（不在 CrossValidation 里）
# 文件夹格式：B22-17_1（末尾 _1=阳性，_0=阴性），标签直接从文件夹名读取，无需 CSV
HOLDOUT_DIR = "HoldOut"

THRESHOLD_FILE = "best_threshold.txt"
OUTPUT_CSV = "holdout_validation_report.csv"

# ==========================================
# 2. 图像预处理（与训练时保持一致）
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

def run_holdout_validation():
    # ------------------------------------------
    # 3. 读取 Phase 3 锁定的阈值
    # ------------------------------------------
    if not os.path.exists(THRESHOLD_FILE):
        raise FileNotFoundError(
            f"找不到 {THRESHOLD_FILE}，请先运行 analyze_results.py（Phase 3）！"
        )
    with open(THRESHOLD_FILE, "r") as f:
        threshold = float(f.read().strip())
    print(f"📐 加载 Phase 3 锁定的最优阈值: {threshold:.1%}")

    # ------------------------------------------
    # 4. 加载模型
    # ------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用设备: {device}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print(f"📦 模型权重加载完毕: {MODEL_PATH}\n")

    # ------------------------------------------
    # 5. 扫描 HoldOut 文件夹
    # ------------------------------------------
    holdout_path = os.path.join(IMG_ROOT, HOLDOUT_DIR)
    all_patient_folders = sorted(glob.glob(os.path.join(holdout_path, "*")))
    all_patient_folders = [p for p in all_patient_folders if os.path.isdir(p)]

    if not all_patient_folders:
        raise RuntimeError(f"HoldOut 文件夹为空或路径有误: {holdout_path}")

    print(f"🔍 HoldOut 共发现 {len(all_patient_folders)} 位病人，开始盲测...")
    print(f"📌 标签来源：文件夹名末尾（_1=阳性，_0=阴性），不依赖任何 CSV\n")

    report = []
    skipped = 0

    for folder in tqdm(all_patient_folders, desc="HoldOut 推理进度"):
        folder_name = os.path.basename(folder)

        # 从文件夹名末尾解析真实标签，例如 B22-17_1 → gt=1
        suffix = folder_name.rsplit('_', 1)[-1]
        if suffix not in ('0', '1'):
            print(f"⚠️  无法解析标签，跳过: {folder_name}")
            skipped += 1
            continue
        gt_label = int(suffix)
        codi = folder_name.rsplit('_', 1)[0]  # 病人 ID，例如 B22-17

        all_imgs = glob.glob(os.path.join(folder, "*.png"))
        if not all_imgs:
            skipped += 1
            continue

        # 批量推理
        ds = FastInferenceDataset(all_imgs)
        dl = DataLoader(ds, batch_size=128, num_workers=4, pin_memory=True)

        pos_count = 0
        with torch.no_grad():
            for batch_imgs in dl:
                batch_imgs = batch_imgs.to(device)
                outputs = model(batch_imgs)
                preds = torch.max(outputs, 1)[1]
                pos_count += torch.sum(preds == 1).item()

        positive_ratio = pos_count / len(all_imgs)
        pred_label = 1 if positive_ratio > threshold else 0

        report.append({
            'Pat_ID': codi,
            'Folder': folder_name,
            'GT_Label': gt_label,
            'Total_Patches': len(all_imgs),
            'Model_Positive_Count': pos_count,
            'Positive_Ratio': positive_ratio,
            'Pred_Label': pred_label,
            'Correct': int(pred_label == gt_label)
        })

    # ------------------------------------------
    # 6. 计算并报告最终结果
    # ------------------------------------------
    report_df = pd.DataFrame(report)
    report_df.to_csv(OUTPUT_CSV, index=False)

    total   = len(report_df)
    correct = report_df['Correct'].sum()
    acc     = correct / total if total > 0 else 0

    pos_df = report_df[report_df['GT_Label'] == 1]
    sens   = (pos_df['Pred_Label'] == 1).mean() if len(pos_df) > 0 else 0

    neg_df = report_df[report_df['GT_Label'] == 0]
    spec   = (neg_df['Pred_Label'] == 0).mean() if len(neg_df) > 0 else 0

    tp = ((report_df['Pred_Label'] == 1) & (report_df['GT_Label'] == 1)).sum()
    fp = ((report_df['Pred_Label'] == 1) & (report_df['GT_Label'] == 0)).sum()
    fn = ((report_df['Pred_Label'] == 0) & (report_df['GT_Label'] == 1)).sum()
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    print("\n" + "="*55)
    print("🎯 Phase 4 HoldOut 盲测验证结果")
    print("="*55)
    print(f"  使用阈值:            {threshold:.1%}  (Phase 3 锁定，从未见过 HoldOut)")
    print(f"  评估病人总数:        {total} 人  (跳过: {skipped})")
    print(f"    - 阳性病人 (GT=1): {len(pos_df)} 人")
    print(f"    - 阴性病人 (GT=0): {len(neg_df)} 人")
    print(f"  ─────────────────────────────────────")
    print(f"  Patient-Level 准确率:  {acc:.2%}  ({correct}/{total})")
    print(f"  敏感度 (Sensitivity):  {sens:.2%}")
    print(f"  特异度 (Specificity):  {spec:.2%}")
    print(f"  F1 Score:              {f1:.2%}")
    print("="*55)
    print(f"\n💾 详细报告已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_holdout_validation()