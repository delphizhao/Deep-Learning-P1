import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms


def simple_test():
    print("--------------------------------------------------")
    print("开始最终调试 (Final Debug)...")

    # 1. 自动定位数据路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(base_dir, 'data', 'HP_WSI-CoordAnnotatedAllPatches.xlsx')
    img_dir = os.path.join(base_dir, 'data', '../data/images')

    # 2. 检查 Excel
    if not os.path.exists(excel_path):
        print(f"错误: 找不到 Excel 文件: {excel_path}")
        return

    print(f"Excel 读取成功")
    df = pd.read_excel(excel_path)

    # 3. 关键诊断：看看你的文件夹里到底有什么
    print("--------------------------------------------------")
    print(f"你的 data/images 文件夹位置: {img_dir}")
    if not os.path.exists(img_dir):
        print("严重错误：images 文件夹不存在！请检查新建文件夹步骤。")
        return

    files_in_dir = os.listdir(img_dir)
    print(f"👀 里面有的前 5 个东西: {files_in_dir[:5]}")
    print("--------------------------------------------------")

    # 4. 开始匹配
    print("正在尝试匹配图片...")
    found_count = 0

    for idx, row in df.iterrows():
        # 获取必要信息
        pat_id = row['Pat_ID']  # 例如 B22-77
        section_id = row['Section_ID']  # 例如 0 或 1
        window_id = row['Window_ID']  # 例如 0

        img_name = f"{window_id}.png"

        # 可能性 A: 图片在子文件夹里 (标准结构: data/images/B22-77_0/0.png)
        folder_name = f"{pat_id}_{section_id}"
        path_a = os.path.join(img_dir, folder_name, img_name)

        # 可能性 B: 图片直接散落在 images 里 (扁平结构: data/images/0.png)
        path_b = os.path.join(img_dir, img_name)

        # 可能性 C: 文件夹名字只有 ID (data/images/B22-77/0.png)
        path_c = os.path.join(img_dir, str(pat_id), img_name)

        final_path = None
        if os.path.exists(path_a):
            final_path = path_a
        elif os.path.exists(path_b):
            final_path = path_b
        elif os.path.exists(path_c):
            final_path = path_c

        if final_path:
            print(f"找到一张! 路径: {final_path}")
            # 测试读取一张就够了，顺便测试 PyTorch
            try:
                img = Image.open(final_path).convert('RGB')
                t = transforms.ToTensor()(img)
                print(f"PyTorch 读取成功，形状: {t.shape}")
                print("\n太棒了！代码和数据终于连通了！")
                return  # 成功退出
            except Exception as e:
                print(f"坏了，文件虽在但读不了: {e}")
                return

    # 如果循环跑完了还没 return，说明一张都没找到
    print("\n匹配失败。")
    print("请看上面的 '你的 data/images 文件夹位置' 和 '里面有的东西'")
    print("确保你下载的图片 (比如 0.png) 确实在那个 Excel 里有记录。")
    print("提示：你可能只下载了 B22-77_0 文件夹，但 Excel 前几行全是 B22-01_1 的数据。")
    print("程序会继续往后扫 Excel，直到找到你下载的那部分数据...")

    # 再次尝试：暴力搜索 Excel 里有没有任何一张图在你文件夹里
    print("\n🔄 正在暴力搜索匹配（可能需要几秒钟）...")
    all_downloaded_files = set(files_in_dir)  # 假设是散落的
    # 如果是文件夹，就看文件夹里的
    for f in files_in_dir:
        sub_path = os.path.join(img_dir, f)
        if os.path.isdir(sub_path):
            print(f"   -> 扫描子文件夹: {f}")
            sub_files = os.listdir(sub_path)
            # 检查 Excel 里有没有这个文件夹的数据
            subset = df[df['Pat_ID'].astype(str) + '_' + df['Section_ID'].astype(str) == f]
            if not subset.empty:
                print(f"  发现 Excel 里有关于文件夹 {f} 的记录！")
                print("   请检查里面图片名字是否匹配，例如 Excel 说有 0.png")
                return
            else:
                print(f"  警告: 你下载了文件夹 {f}，但 Excel 里好像没有这个 ID 的记录？")


if __name__ == '__main__':
    simple_test()