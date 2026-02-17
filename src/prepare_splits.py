import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                   help="Root dir containing patient folders like B22-83_1/0.png")
    p.add_argument("--xlsx", type=str, default="data/meta/HP_WSI-CoordAnnotatedAllPatches.xlsx",
                   help="Excel with patch annotations: Pat_ID, Section_ID, Window_ID, Presence")
    p.add_argument("--patient_csv", type=str, default="data/meta/PatientDiagnosis.csv",
                   help="PatientDiagnosis.csv: CODI + diagnosis (NEGATIVA/BAIXA/ALTA)")
    p.add_argument("--out_dir", type=str, default="splits", help="Output directory")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)

    # for quick debug only
    p.add_argument("--limit_patients", type=int, default=0,
                   help="If >0, only use this many patients (debug)")
    return p.parse_args()


def patient_stratified_split(diag_df: pd.DataFrame, train_r: float, val_r: float, test_r: float, seed: int):
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "train+val+test must sum to 1.0"
    rng = np.random.default_rng(seed)

    train_ids, val_ids, test_ids = [], [], []
    for label in sorted(diag_df["patient_label"].unique()):
        sub = diag_df[diag_df["patient_label"] == label].copy()
        ids = sub["patient_id"].tolist()
        rng.shuffle(ids)

        n = len(ids)
        n_train = int(round(n * train_r))
        n_val = int(round(n * val_r))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        train_ids += ids[:n_train]
        val_ids += ids[n_train:n_train + n_val]
        test_ids += ids[n_train + n_val:]

        print(f"[INFO] patient_label={label}: total={n} -> train={n_train}, val={n_val}, test={n_test}")

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)
    return train_ids, val_ids, test_ids


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    required = {"Pat_ID", "Section_ID", "Window_ID", "Presence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel missing columns: {missing}")

    # keep labeled only
    df = df[df["Presence"].isin([1, -1])].copy()
    df["label"] = (df["Presence"] == 1).astype(int)

    # Cropped folder structure: Pat_ID_SectionID/Window_ID.png
    def make_rel_path(row):
        folder = f"{str(row['Pat_ID']).strip()}_{int(float(row['Section_ID']))}"

        wid = str(row["Window_ID"]).strip()
        # 如果 Excel 里已经带 .png，就别重复加
        if wid.lower().endswith(".png"):
            fname = wid
        else:
            fname = f"{wid}.png"

        return str(Path(folder) / fname)

    df["rel_path"] = df.apply(make_rel_path, axis=1)
    df["path"] = df["rel_path"].apply(lambda p: str(data_root / p))

    df["patient_id"] = df["Pat_ID"].astype(str)
    df["section_id"] = df["Section_ID"].astype(int)
    df["window_id"] = df["Window_ID"].astype(int)
    df["source"] = "annotated_excel"

    print(f"[INFO] labeled patches: {len(df)}")
    print(f"[INFO] patch patients: {df['patient_id'].nunique()}")

    patient_csv = Path(args.patient_csv)
    if not patient_csv.exists():
        raise FileNotFoundError(f"PatientDiagnosis.csv not found: {patient_csv}")

    diag = pd.read_csv(patient_csv)
    if "CODI" not in diag.columns:
        raise ValueError("PatientDiagnosis.csv must contain column 'CODI'")

    diag = diag.rename(columns={"CODI": "patient_id"})
    diag["patient_id"] = diag["patient_id"].astype(str)
    diag["patient_label"] = diag.iloc[:, 1].map(lambda x: 0 if str(x).upper().startswith("NEG") else 1)
    diag = diag[["patient_id", "patient_label"]].drop_duplicates()

    # intersection only
    diag = diag[diag["patient_id"].isin(df["patient_id"].unique())].copy()
    print(f"[INFO] patients available for split: {diag['patient_id'].nunique()}")

    if args.limit_patients and args.limit_patients > 0:
        diag = diag.sample(n=min(args.limit_patients, len(diag)), random_state=args.seed).copy()
        df = df[df["patient_id"].isin(diag["patient_id"])].copy()
        print(f"[DEBUG] limit_patients={args.limit_patients} -> patches={len(df)}")

    train_ids, val_ids, test_ids = patient_stratified_split(diag, args.train, args.val, args.test, args.seed)

    train_df = df[df["patient_id"].isin(train_ids)].copy()
    val_df = df[df["patient_id"].isin(val_ids)].copy()
    test_df = df[df["patient_id"].isin(test_ids)].copy()

    # sample existence check (fast)
    sample_paths = pd.concat([train_df, val_df, test_df]).sample(n=min(50, len(df)), random_state=args.seed)["path"].tolist()
    exist_cnt = sum(Path(p).exists() for p in sample_paths)
    print(f"[INFO] sampled path existence: {exist_cnt}/{len(sample_paths)}")

    cols = ["path", "label", "patient_id", "section_id", "window_id", "source"]
    (out_dir / "train.csv").write_text(train_df[cols].to_csv(index=False), encoding="utf-8")
    (out_dir / "val.csv").write_text(val_df[cols].to_csv(index=False), encoding="utf-8")
    (out_dir / "test.csv").write_text(test_df[cols].to_csv(index=False), encoding="utf-8")

    train_neg_df = train_df[train_df["label"] == 0].copy()
    (out_dir / "train_neg.csv").write_text(train_neg_df[cols].to_csv(index=False), encoding="utf-8")

    def report(name, d):
        pos = int((d["label"] == 1).sum())
        neg = int((d["label"] == 0).sum())
        print(f"[INFO] {name}: patches={len(d)} pos={pos} neg={neg} patients={d['patient_id'].nunique()}")

    report("train", train_df)
    report("val", val_df)
    report("test", test_df)
    print(f"[OK] wrote splits to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
