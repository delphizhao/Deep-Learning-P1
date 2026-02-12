import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import os
from dataset import HPyloriDataset


def main():
    print("--- 🔬 幽门螺杆菌 AI 训练+验证模式启动 ---")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(base_dir, 'data', 'HP_WSI-CoordAnnotatedAllPatches.xlsx')
    img_dir = os.path.join(base_dir, 'data', 'images')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. 加载全量数据集
    full_dataset = HPyloriDataset(excel_path, img_dir, transform=transform, local_debug=True)

    # 2. 划分数据集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    # 注意：本地只有5张图时，train可能4张，val可能1张
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print(f"📊 划分完成: 训练集 {len(train_dataset)} 张, 验证集 {len(val_dataset)} 张")

    # 3. 初始化模型
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 4. 训练循环
    for epoch in range(5):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- 验证阶段 (这就是你要的步骤) ---
        model.eval()  # 切换到评估模式
        correct = 0
        total = 0
        with torch.no_grad():  # 考试时不需要记录梯度，节省内存
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/5] | 训练 Loss: {train_loss / len(train_loader):.4f} | 验证准确率: {accuracy:.2f}%")

    print("\n🎉 训练与验证逻辑测试完成！")


if __name__ == '__main__':
    main()