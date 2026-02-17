import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                   help="Root directory containing patient folders like B22-83_1/0.png")
    p.add_argument("--xlsx", type=str, default="data/meta/HP_WSI-CoordAnnotatedAllPatches.xlsx",
                   help="Excel with patch annotations (Presence, Pat_ID, Section_ID, Window_ID)")
    p.add_argument("--patient_csv", type=str, default="data/meta/PatientDiagnosis.csv",
                   help="CSV with patient diagnosis (CODI + NEGATIVA/BAIXA/ALTA)")
    p.add_argument("--out_dir", type=str, default="splits",
                   help="Output directory for split CSVs")
    p.add_argument("--seed", type=int, default=42)

    # split ratios (patient-level)
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)

    # optional: limit patients for quick debug
    p.add_argument("--limit_patients", type=int, default=0,
                   help="If >0, only use this many patients (debug only).")
    return p.parse_args()


def patient_stratified_split(diag_df: pd.DataFrame, train_r: float, val_r: float, test_r: float, seed: int):
    """
    Split patients into train/val/test with stratification by patient_label.
    diag_df columns: patient_id, patient_label
    """
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "train+val+test must sum to 1"

    rng = np.random.default_rng(seed)

    train_ids, val_ids, test_ids = [], [], []

    for label in sorted(diag_df["patient_label"].unique()):
        sub = diag_df[diag_df["patient_label"] == label].copy()
        ids = sub["patient_id"].tolist()
        rng.shuffle(ids)

        n = len(ids)
        n_train = int(round(n * train_r))
        n_val = int(round(n * val_r))
        # ensure total == n
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        train_ids += ids[:n_train]
        val_ids += ids[n_train:n_train + n_val]
        test_ids += ids[n_train + n_val:]

        print(f"[INFO] label={label} patients={n} -> train={n_train}, val={n_val}, test={n_test}")

    # shuffle overall lists (optional)
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)
    return train_ids, val_ids, test_ids


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) read patch annotations
    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel not found: {xlsx_path}")
    df = pd.read_excel(xlsx_path)

    required_cols = {"Pat_ID", "Section_ID", "Window_ID", "Presence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Excel missing columns: {missing}")

    # keep only valid labels: Presence in {1,-1}
    df = df[df["Presence"].isin([1, -1])].copy()
    df["label"] = (df["Presence"] == 1).astype(int)

    # build rel_path that matches Cropped structure in your repo/server:
    # data_root / (Pat_ID_Section_ID) / (Window_ID.png)
    def make_rel_path(row):
        folder = f"{row['Pat_ID']}_{int(row['Section_ID'])}"
        fname = f"{int(row['Window_ID'])}.png"
        return str(Path(folder) / fname)

    df["rel_path"] = df.apply(make_rel_path, axis=1)
    df["path"] = df["rel_path"].apply(lambda p: str(data_root / p))

    df["patient_id"] = df["Pat_ID"].astype(str)
    df["section_id"] = df["Section_ID"].astype(int)
    df["window_id"] = df["Window_ID"].astype(int)
    df["source"] = "annotated_excel"

    print(f"[INFO] labeled patches (Presence in {{1,-1}}): {len(df)}")
    print(f"[INFO] unique patients in patches: {df['patient_id'].nunique()}")

    # 2) read patient diagnosis for patient-level stratification
    patient_csv = Path(args.patient_csv)
    if not patient_csv.exists():
        raise FileNotFoundError(f"PatientDiagnosis.csv not found: {patient_csv}")
    diag = pd.read_csv(patient_csv)

    if "CODI" not in diag.columns:
        raise ValueError("PatientDiagnosis.csv must contain column 'CODI'")

    diag = diag.rename(columns={"CODI": "patient_id"})
    # map diagnosis to binary patient_label
    # NEGATIVA -> 0, BAIXA/ALTA -> 1
    diag["patient_id"] = diag["patient_id"].astype(str)
    diag["patient_label"] = diag.iloc[:, 1].map(lambda x: 0 if str(x).upper().startswith("NEG") else 1)

    # Keep only patients that appear in patch df
    diag = diag[diag["patient_id"].isin(df["patient_id"].unique())].copy()
    diag = diag[["patient_id", "patient_label"]].drop_duplicates()

    print(f"[INFO] patients available for splitting (intersection): {diag['patient_id'].nunique()}")

    # optional debug limit
    if args.limit_patients and args.limit_patients > 0:
        diag = diag.sample(n=min(args.limit_patients, len(diag)), random_state=args.seed).copy()
        df = df[df["patient_id"].isin(diag["patient_id"])].copy()
        print(f"[DEBUG] limit_patients={args.limit_patients} -> patches={len(df)} patients={diag['patient_id'].nunique()}")

    # 3) patient-level split
    train_ids, val_ids, test_ids = patient_stratified_split(
        diag, args.train, args.val, args.test, args.seed
    )

    # 4) filter patch rows by patient split
    train_df = df[df["patient_id"].isin(train_ids)].copy()
    val_df = df[df["patient_id"].isin(val_ids)].copy()
    test_df = df[df["patient_id"].isin(test_ids)].copy()

    # sanity: no overlap
    assert set(train_df["patient_id"]).isdisjoint(set(val_df["patient_id"]))
    assert set(train_df["patient_id"]).isdisjoint(set(test_df["patient_id"]))
    assert set(val_df["patient_id"]).isdisjoint(set(test_df["patient_id"]))

    # 5) quick path sanity check (sample a few)
    sample_paths = pd.concat([train_df, val_df, test_df]).sample(n=min(20, len(df)), random_state=args.seed)["path"].tolist()
    exist_cnt = sum(Path(p).exists() for p in sample_paths)
    print(f"[INFO] sampled path existence check: {exist_cnt} / {len(sample_paths)} exist (sample only)")

    # 6) write CSVs
    cols = ["path", "label", "patient_id", "section_id", "window_id", "source"]
    train_csv = out_dir / "train.csv"
    val_csv = out_dir / "val.csv"
    test_csv = out_dir / "test.csv"

    train_df[cols].to_csv(train_csv, index=False)
    val_df[cols].to_csv(val_csv, index=False)
    test_df[cols].to_csv(test_csv, index=False)

    # AE training: negative-only from train split
    train_neg_df = train_df[train_df["label"] == 0].copy()
    train_neg_csv = out_dir / "train_neg.csv"
    train_neg_df[cols].to_csv(train_neg_csv, index=False)

    # report
    def report(name, d):
        pos = int((d["label"] == 1).sum())
        neg = int((d["label"] == 0).sum())
        print(f"[INFO] {name}: patches={len(d)} pos={pos} neg={neg} patients={d['patient_id'].nunique()}")

    report("train", train_df)
    report("val", val_df)
    report("test", test_df)
    print(f"[OK] wrote: {train_csv}")
    print(f"[OK] wrote: {val_csv}")
    print(f"[OK] wrote: {test_csv}")
    print(f"[OK] wrote: {train_neg_csv}")


if __name__ == "__main__":
    main()
