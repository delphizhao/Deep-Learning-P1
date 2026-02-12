import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class HPyloriDataset(Dataset):
    def __init__(self, excel_path, img_root_dir, transform=None, local_debug=True):
        """
        :param excel_path: Excel 文件路径
        :param img_root_dir: 图片存储根目录 (data/images)
        :param transform: 图像预处理变换
        :param local_debug: 如果为 True，将只保留本地硬盘里确实存在的图片，防止训练崩溃
        """
        self.img_root_dir = img_root_dir
        self.transform = transform

        # 1. 加载 Excel
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"❌ 找不到 Excel 文件: {excel_path}")

        print(f"📖 正在读取索引文件...")
        df = pd.read_excel(excel_path)

        # 2. 标签预处理: 只保留 1 (阳性) 和 -1 (阴性)
        # 顺便把 -1 映射为 0，因为 PyTorch 的分类标签通常要求从 0 开始
        if 'Presence' in df.columns:
            df = df[df['Presence'].isin([1, -1])].copy()
            df['label'] = df['Presence'].apply(lambda x: 1 if x == 1 else 0)
        else:
            raise ValueError("❌ Excel 中缺少必要的 'Presence' 列")

        # 3. 本地调试模式：过滤掉没下载的图片
        if local_debug:
            print("🔍 本地调试模式：正在扫描硬盘，剔除未下载的样本...")
            valid_mask = []
            for _, row in df.iterrows():
                # 尝试匹配你目前的扁平化路径 (直接放在 images 下)
                img_path = os.path.join(self.img_root_dir, f"{row['Window_ID']}.png")
                # 如果未来你用了文件夹结构，可以增加判断：
                # folder_path = os.path.join(self.img_root_dir, f"{row['Pat_ID']}_{row['Section_ID']}", f"{row['Window_ID']}.png")
                valid_mask.append(os.path.exists(img_path))

            df = df[valid_mask].reset_index(drop=True)
            print(f"✅ 扫描完成！本地可用样本数: {len(df)}")
        else:
            df = df.reset_index(drop=True)
            print(f"🚀 全量模式：总样本数: {len(df)}")

        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 获取图片 ID
        window_id = row['Window_ID']
        label = row['label']

        # 拼接图片路径
        # 注意：这里优先匹配你目前拖进 images 文件夹的扁平结构
        img_name = f"{window_id}.png"
        img_path = os.path.join(self.img_root_dir, img_name)

        # 如果主路径找不到，尝试子文件夹结构 (为了兼容服务器)
        if not os.path.exists(img_path):
            folder_name = f"{row['Pat_ID']}_{row['Section_ID']}"
            img_path = os.path.join(self.img_root_dir, folder_name, img_name)

        try:
            # 读取并转为 RGB (防止有灰度图干扰)
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 极端情况：如果文件损坏或丢失，返回一张黑图占位
            print(f"⚠️ 读取失败: {img_path}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------
# 下面这段代码只有当你直接运行 python dataset.py 时才会执行，用于快速自检
if __name__ == '__main__':
    print("🧪 正在自检 dataset.py...")
    # 这里的路径根据你的 PyCharm 结构自动推断
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_excel = os.path.join(base, 'data', 'HP_WSI-CoordAnnotatedAllPatches.xlsx')
    test_imgs = os.path.join(base, 'data', 'images')

    try:
        ds = HPyloriDataset(test_excel, test_imgs, local_debug=True)
        if len(ds) > 0:
            img, lbl = ds[0]
            print(f"✅ 自检成功！第一张图尺寸: {img.size}, 标签: {lbl}")
        else:
            print("⚠️ 警告：没找到任何本地图片，请检查 data/images 文件夹。")
    except Exception as e:
        print(f"❌ 自检失败: {e}")