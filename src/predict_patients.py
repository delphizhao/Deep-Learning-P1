import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# 1. 核心路径配置
# ==========================================
MODEL_PATH = "/import2/hhome/ricse05/Deep-Learning-P1/runs/resnet18_seed42/best.pth"
PATIENT_CSV = "/fhome/vlia/HelicoDataSet/PatientDiagnosis.csv"
IMG_ROOT = "/fhome/vlia/HelicoDataSet"

# ⚠️ Phase 2 改动：只扫描 Cropped，完全不碰 HoldOut
SEARCH_DIRS = ["CrossValidation/Cropped"]

OUTPUT_CSV = "cropped_patient_diagnosis_report.csv"  # 输出文件名也改了，避免覆盖旧结果

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

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用设备: {device}")

    # ------------------------------------------
    # 3. 加载模型
    # ------------------------------------------
    print(f"📦 加载权重文件: {MODEL_PATH}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        print("💡 检测到大礼包格式，正在提取 model_state...")
        model.load_state_dict(checkpoint['model_state'])
    else:
        print("💡 检测到直接权重格式，正在加载...")
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # ------------------------------------------
    # 4. 准备病人名单（只处理能在 Cropped 里找到图片的病人）
    # ------------------------------------------
    patient_df = pd.read_csv(PATIENT_CSV)
    final_report = []
    skipped = 0

    print(f"🔍 开始为 {len(patient_df)} 位病人扫描 Cropped 文件夹...")
    print(f"⚠️  注意：HoldOut 文件夹在本阶段完全跳过，留给 Phase 4 盲测。\n")

    for _, row in tqdm(patient_df.iterrows(), total=len(patient_df), desc="推理进度"):
        codi = str(row['CODI'])

        # 只在 Cropped 里找病人文件夹
        patient_folder = None
        for d in SEARCH_DIRS:
            matches = glob.glob(os.path.join(IMG_ROOT, d, f"{codi}_*"))
            if matches:
                patient_folder = matches[0]
                break

        if not patient_folder:
            skipped += 1
            continue  # 不在 Cropped 里的病人（即 HoldOut 病人）直接跳过

        all_imgs = glob.glob(os.path.join(patient_folder, "*.png"))
        if not all_imgs:
            skipped += 1
            continue

        # ------------------------------------------
        # 5. 批量推理该病人的所有切片
        # ------------------------------------------
        ds = FastInferenceDataset(all_imgs)
        dl = DataLoader(ds, batch_size=128, num_workers=4, pin_memory=True)

        pos_count = 0
        with torch.no_grad():
            for batch_imgs in dl:
                batch_imgs = batch_imgs.to(device)
                outputs = model(batch_imgs)
                preds = torch.max(outputs, 1)[1]
                pos_count += torch.sum(preds == 1).item()

        final_report.append({
            'Pat_ID': codi,
            'Doctor_Diagnosis': row['DENSITAT'],
            'Total_Patches': len(all_imgs),
            'Model_Positive_Count': pos_count,
            'Positive_Ratio': pos_count / len(all_imgs),
            'Source': 'Cropped'  # 新增字段，方便后续追踪数据来源
        })

    # ------------------------------------------
    # 6. 保存结果
    # ------------------------------------------
    report_df = pd.DataFrame(final_report)
    report_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n" + "="*55)
    print(f"✅ Phase 2 推理完成（仅 Cropped）")
    print(f"📊 成功推理病人数: {len(final_report)}")
    print(f"⏭️  跳过病人数（不在 Cropped / 无图片）: {skipped}")
    print(f"💾 结果已保存至: {OUTPUT_CSV}")
    print(f"➡️  下一步：运行 analyze_results.py 寻找最优阈值")
    print("="*55)

if __name__ == "__main__":
    run_inference()