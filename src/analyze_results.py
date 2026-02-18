import pandas as pd

# ==========================================
# 配置
# ==========================================
INPUT_CSV = "cropped_patient_diagnosis_report.csv"  # Phase 2 的输出

# ==========================================
# 1. 读取 Phase 2 的推理报告（全是 Cropped 病人）
# ==========================================
df = pd.read_csv(INPUT_CSV)

# 过滤掉没有医生标注的病人（Doctor_Diagnosis 为空或未知）
df = df[df['Doctor_Diagnosis'].notna()].copy()

# 转换标签：NEGATIVA → 0，其他(BAIXA/MODERADA/ALTA) → 1
df['GT_Label'] = df['Doctor_Diagnosis'].apply(lambda x: 0 if str(x).strip() == 'NEGATIVA' else 1)

print(f"📋 用于阈值分析的病人总数: {len(df)}")
print(f"   - 阳性（有感染）: {df['GT_Label'].sum()} 人")
print(f"   - 阴性（无感染）: {(df['GT_Label'] == 0).sum()} 人")
print(f"   - 数据来源: 仅 Cropped（HoldOut 未参与）\n")

# ==========================================
# 2. 遍历阈值，寻找最优
# ==========================================
print(f"{'阈值':<10} | {'总准确率':<10} | {'敏感度(Recall)':<14} | {'特异度(Spec)':<12} | {'F1 Score'}")
print("-" * 68)

best_acc = 0
best_threshold = None
best_row = None
results = []

for t in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
    df['Pred'] = (df['Positive_Ratio'] > t).astype(int)

    acc  = (df['Pred'] == df['GT_Label']).mean()

    pos_df = df[df['GT_Label'] == 1]
    sens = (pos_df['Pred'] == 1).mean() if len(pos_df) > 0 else 0

    neg_df = df[df['GT_Label'] == 0]
    spec = (neg_df['Pred'] == 0).mean() if len(neg_df) > 0 else 0

    # F1 Score
    tp = ((df['Pred'] == 1) & (df['GT_Label'] == 1)).sum()
    fp = ((df['Pred'] == 1) & (df['GT_Label'] == 0)).sum()
    fn = ((df['Pred'] == 0) & (df['GT_Label'] == 1)).sum()
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    marker = " ◀ 当前最优" if acc > best_acc else ""
    print(f"{t:10.1%} | {acc:10.2%} | {sens:14.2%} | {spec:12.2%} | {f1:8.2%}{marker}")

    if acc > best_acc:
        best_acc = acc
        best_threshold = t
        best_row = {'threshold': t, 'accuracy': acc, 'sensitivity': sens, 'specificity': spec, 'f1': f1}

    results.append({'threshold': t, 'accuracy': acc, 'sensitivity': sens, 'specificity': spec, 'f1': f1})

# ==========================================
# 3. 保存最优阈值供 Phase 4 使用
# ==========================================
print("\n" + "="*55)
print(f"🏆 最优阈值: {best_threshold:.1%}")
print(f"   准确率:  {best_row['accuracy']:.2%}")
print(f"   敏感度:  {best_row['sensitivity']:.2%}")
print(f"   特异度:  {best_row['specificity']:.2%}")
print(f"   F1 Score: {best_row['f1']:.2%}")
print("="*55)

# 将最优阈值写入文件，供 Phase 4 脚本自动读取
with open("best_threshold.txt", "w") as f:
    f.write(str(best_threshold))
print(f"\n💾 最优阈值已保存至 best_threshold.txt，供 Phase 4 自动加载。")
print(f"➡️  下一步：运行 holdout_validation.py 进行盲测验证")

# 保存完整阈值分析表
results_df = pd.DataFrame(results)
results_df.to_csv("threshold_analysis_cropped.csv", index=False)
print(f"📊 完整阈值分析表已保存至 threshold_analysis_cropped.csv")