import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import os
import sys

# 引入刚才写的 dataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import HPyloriDataset

# ================= 配置区域 =================
# ⚠️⚠️⚠️ 关键开关 ⚠️⚠️⚠️
# 在本地跑测试时，设为 True (只跑几张图)
# 上传到服务器前，改成 False (跑 21 万张图)
IS_LOCAL = False


# ===========================================

def main():
    print(f"--- 🚀 训练启动 (模式: {'本地调试' if IS_LOCAL else '服务器全量'}) ---")

    # 1. 自动检测显卡
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 使用设备: {device}")

    # 2. 路径设置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(base_dir, 'data', 'HP_WSI-CoordAnnotatedAllPatches.xlsx')
    img_dir = os.path.join(base_dir, 'data', 'images')

    # 3. 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. 加载数据集
    try:
        full_dataset = HPyloriDataset(excel_path, img_dir, transform=transform, local_debug=IS_LOCAL)
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    if len(full_dataset) == 0:
        print("❌ 错误：未找到有效样本，请检查路径。")
        return

    # 5. 划分 80% 训练, 20% 验证
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 根据模式自动调整参数
    batch_size = 4 if IS_LOCAL else 32  # 服务器显卡好，一次吃 32 张
    num_workers = 0 if IS_LOCAL else 4  # 服务器 CPU 强，开 4 个进程加速读取

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"📊 准备完毕: 训练集 {len(train_dataset)} 张, 验证集 {len(val_dataset)} 张")

    # 6. 搭建模型 (ResNet18)
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2分类: 有菌/无菌
    model = model.to(device)

    # 7. 优化器与损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 8. 训练循环
    epochs = 2 if IS_LOCAL else 10  # 本地跑2轮尝尝鲜，服务器跑10轮动真格
    best_acc = 0.0

    for epoch in range(epochs):
        print(f"\nepoch {epoch + 1}/{epochs} 开始...")

        # --- 训练 ---
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 服务器上每 100 个 batch 报一次平安
            if not IS_LOCAL and (i + 1) % 100 == 0:
                print(f"   Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)

        # --- 验证 ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"🏁 Epoch [{epoch + 1}/{epochs}] 结束 | 训练 Loss: {epoch_loss:.4f} | 验证准确率: {accuracy:.2f}%")

        # --- 保存最好的模型 ---
        if accuracy > best_acc:
            best_acc = accuracy
            save_path = "best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 发现新纪录！模型已保存至: {save_path}")

    print("\n🎉 全流程结束！")


if __name__ == '__main__':
    main()