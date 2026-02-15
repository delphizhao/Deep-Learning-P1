import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class HPyloriDataset(Dataset):
    def __init__(self, excel_path, img_root_dir, transform=None, local_debug=True):
        """
        :param excel_path: Excel 索引文件路径
        :param img_root_dir: 图片文件夹根目录
        :param local_debug:
            True = 本地模式（只加载硬盘里有的几张图，适合调试）
            False = 服务器模式（加载 Excel 里所有有标注的图，约21万张）
        """
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.local_debug = local_debug

        # 1. 读取 Excel
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"❌ 找不到 Excel 文件: {excel_path}")

        print(f"📖 正在读取索引文件: {excel_path} ...")
        df = pd.read_excel(excel_path)

        # 2. 核心过滤：只保留有明确标注的行 (1=有菌, -1=无菌)
        # 这一步解决了“没有答案训练不起来”的问题
        if 'Presence' in df.columns:
            df = df[df['Presence'].isin([1, -1])].copy()
            # 将 -1 (无菌) 转换为 0，1 (有菌) 保持为 1
            df['label'] = df['Presence'].apply(lambda x: 1 if x == 1 else 0)
        else:
            raise ValueError("❌ Excel 中缺少 'Presence' 列，无法训练！")

        # 3. 本地调试逻辑
        if local_debug:
            print("🔍 [本地模式] 正在扫描硬盘，剔除未下载的图片...")
            valid_rows = []
            for _, row in df.iterrows():
                # 检查图片是否存在（支持两种常见的路径结构）
                if self._check_path(row):
                    valid_rows.append(row)

            df = pd.DataFrame(valid_rows).reset_index(drop=True)
            print(f"✅ [本地模式] 过滤完成，实际可用样本数: {len(df)}")
        else:
            # 服务器模式：直接信任 Excel，不再逐一检查硬盘（为了速度）
            df = df.reset_index(drop=True)
            print(f"🚀 [服务器模式] 加载全量数据，计划训练样本数: {len(df)}")

        self.data = df

    def _check_path(self, row):
        """辅助函数：检查图片路径是否存在"""
        # 尝试路径 1: data/images/Window_ID.png (扁平结构)
        path1 = os.path.join(self.img_root_dir, f"{row['Window_ID']}.png")
        if os.path.exists(path1): return True

        # 尝试路径 2: data/images/Pat_ID_Section_ID/Window_ID.png (层级结构)
        folder_name = f"{row['Pat_ID']}_{row['Section_ID']}"
        path2 = os.path.join(self.img_root_dir, folder_name, f"{row['Window_ID']}.png")
        if os.path.exists(path2): return True

        return False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = row['label']
        img_name = f"{row['Window_ID']}.png"

        # 动态寻找图片路径 (优先找扁平结构，再找文件夹结构)
        img_path = os.path.join(self.img_root_dir, img_name)
        if not os.path.exists(img_path):
            folder_name = f"{row['Pat_ID']}_{row['Section_ID']}"
            img_path = os.path.join(self.img_root_dir, folder_name, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # 万一图片损坏，返回一张全黑图片防止训练中断
            # print(f"⚠️ 图片读取失败: {img_path}") # 只有调试时才打开这个打印
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)