import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 这一行是为了确保能引用到你刚才写的 dataset.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import HPyloriDataset  # <--- 引用你刚才写的类


def test_my_dataloader():
    print("🚀 开始测试数据加载器 (DataLoader Test)...")

    # 1. 自动定位路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # src 的上一级

    excel_path = os.path.join(project_root, 'data', 'HP_WSI-CoordAnnotatedAllPatches.xlsx')
    img_dir = os.path.join(project_root, 'data', 'images')

    print(f"📂 数据目录: {img_dir}")

    # 2. 定义简单的图片预处理
    # 我们把所有图片都缩放到 224x224，这是深度学习的标准尺寸
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 3. 初始化数据集
    try:
        my_dataset = HPyloriDataset(
            excel_path=excel_path,
            img_root_dir=img_dir,
            transform=data_transform
        )
        print(f"✅ Dataset 初始化成功! 样本总数: {len(my_dataset)}")
    except Exception as e:
        print(f"❌ Dataset 初始化失败: {e}")
        return

    # 4. 初始化 DataLoader (关键步骤)
    # batch_size=4 意味着一次拿 4 张图
    loader = DataLoader(my_dataset, batch_size=4, shuffle=True)

    # 5. 尝试拿出一个 Batch 看看
    print("\n🔄 正在尝试读取一个 Batch (4张图)...")
    try:
        # iter(loader) 创建迭代器，next() 拿第一组数据
        images, labels = next(iter(loader))

        print("\n🎉 成功！DataLoader 工作正常！")
        print("-" * 30)
        print(f"🖼️ 图片 Batch 形状: {images.shape}")
        print("   -> [4, 3, 224, 224] 分别代表: [4张图, 3个颜色通道, 高224, 宽224]")
        print(f"🏷️ 标签 Batch 形状: {labels.shape}")
        print(f"🔢 具体标签值: {labels}")
        print("-" * 30)

    except Exception as e:
        print(f"❌ 读取 Batch 失败: {e}")
        print("可能原因：")
        print("1. dataset.py 里的 __getitem__ 逻辑有 bug")
        print("2. 图片路径拼接不对，导致找不到文件")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_my_dataloader()