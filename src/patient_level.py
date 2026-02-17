import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def patient_level_aggregation(
    patch_csv,
    prob_col="prob",      # patch预测为正的概率
    label_col="label",    # patch GT (只在 CV set 有)
    patient_col="Pat_ID",
    patient_label_csv=None,  # PatientDiagnosis.csv
    threshold=0.05
):
    """
    threshold: 病人被判为 positive 的 patch 比例阈值
    """

    df = pd.read_csv(patch_csv)

    # 按病人聚合
    patient_df = df.groupby(patient_col).agg(
        pos_ratio=(prob_col, lambda x: (x > 0.5).mean()),
        max_prob=(prob_col, "max"),
        mean_prob=(prob_col, "mean")
    ).reset_index()

    # 用比例规则判断病人
    patient_df["pred_patient"] = (patient_df["pos_ratio"] >= threshold).astype(int)

    # 如果有 GT（CrossValidation set）
    if patient_label_csv is not None:
        gt = pd.read_csv(patient_label_csv)
        patient_df = patient_df.merge(
            gt[[patient_col, "Diagnosis"]],
            on=patient_col,
            how="inner"
        )

        acc = accuracy_score(
            patient_df["Diagnosis"],
            patient_df["pred_patient"]
        )

        auc = roc_auc_score(
            patient_df["Diagnosis"],
            patient_df["mean_prob"]
        )

        print(f"[Patient-level] ACC={acc:.4f}  AUC={auc:.4f}")

    return patient_df
