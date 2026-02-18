import argparse
import pandas as pd
from pathlib import Path
import numpy as np
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder of cropped images, e.g. CrossValidation/Cropped",
    )
    parser.add_argument(
        "--xlsx",
        type=str,
        required=True,
        help="Annotation Excel file",
    )
    parser.add_argument(
        "--patient_csv",
        type=str,
        default="data/PatientDiagnosis.csv",
        help="Patient-level diagnosis CSV (optional)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="splits",
        help="Output directory for split CSVs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict_paths",
        action="store_true",
        help="Only trust canonical {Pat_ID}_{Section_ID}/{Window_ID}.png paths (disable fallback search).",
    )
    return parser.parse_args()


# -------------------------
# build relative image path
# -------------------------
def make_rel_path(row):
    pat = str(row["Pat_ID"]).strip()
    sec_raw = str(row["Section_ID"]).strip()
    sec = sec_raw[:-2] if sec_raw.endswith(".0") else sec_raw
    wid = str(row["Window_ID"]).strip()

    # 去掉 .0
    if wid.endswith(".0"):
        wid = wid[:-2]

    # 确保 .png
    if not wid.lower().endswith(".png"):
        wid = f"{wid}.png"

    return f"{pat}_{sec}/{wid}"


def build_file_index(data_root: Path):
    """Index png files to improve hit-rate when folder layout differs from canonical rel_path."""
    by_rel = {}
    by_basename = defaultdict(list)

    for p in data_root.rglob("*.png"):
        rel = str(p.relative_to(data_root)).replace("\\", "/")
        by_rel[rel] = p
        by_basename[p.name].append(p)

    return by_rel, by_basename


def resolve_row_path(row, data_root: Path, by_rel, by_basename) -> str | None:
    """Try canonical path first, then fallback by basename if unique."""
    rel_path = row["rel_path"]
    canonical = data_root / rel_path
    if canonical.exists():
        return str(canonical)

    if rel_path in by_rel:
        return str(by_rel[rel_path])

    name = Path(rel_path).name
    matches = by_basename.get(name, [])
    if len(matches) == 1:
        return str(matches[0])

    return None


# -------------------------
# patch-level split fallback
# -------------------------
def patch_level_split(df, seed=42, train_ratio=0.7, val_ratio=0.15):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    return (
        df.iloc[train_idx].copy(),
        df.iloc[val_idx].copy(),
        df.iloc[test_idx].copy(),
    )


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    xlsx_path = Path(args.xlsx)
    patient_csv = Path(args.patient_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # load Excel
    # -------------------------
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel not found: {xlsx_path}")

    print(f"[INFO] Reading Excel: {xlsx_path}")
    df = pd.read_excel(xlsx_path)

    # -------------------------
    # keep labeled patches only
    # -------------------------
    if "Presence" not in df.columns:
        raise ValueError("Excel missing column: Presence")

    df = df[df["Presence"].isin([1, -1])].copy()
    df["label"] = (df["Presence"] == 1).astype(int)

    print(f"[INFO] labeled patches: {len(df)}")

    # -------------------------
    # build paths
    # -------------------------
    df["rel_path"] = df.apply(make_rel_path, axis=1)

    if args.strict_paths:
        df["path"] = df["rel_path"].apply(lambda p: str(data_root / p))
    else:
        print("[INFO] building recursive image index (for robust path matching)...")
        by_rel, by_basename = build_file_index(data_root)
        print(f"[INFO] indexed png files: {len(by_rel)}")
        df["path"] = df.apply(
            lambda row: resolve_row_path(row, data_root, by_rel, by_basename), axis=1
        )

    # check existence
    exists = df["path"].apply(lambda p: isinstance(p, str) and Path(p).exists())
    print(f"[INFO] resolved existing image files: {exists.sum()} / {len(df)}")

    if exists.sum() == 0:
        print("[WARN] 0 images found. Check --data_root")
        print("[WARN] example rel_path:", df["rel_path"].iloc[0])
    else:
        missing = df.loc[~exists, "rel_path"].head(5).tolist()
        if missing:
            print("[INFO] example unresolved rel_paths:")
            for m in missing:
                print(" -", m)

    df = df[exists].reset_index(drop=True)

    # -------------------------
    # try patient-level CSV
    # -------------------------
    if not patient_csv.exists():
        print(f"[WARN] PatientDiagnosis.csv not found: {patient_csv}")
        print("[WARN] Fallback to PATCH-LEVEL split")
        diag = None
    else:
        print(f"[INFO] Reading patient CSV: {patient_csv}")
        diag = pd.read_csv(patient_csv)

        if "CODI" not in diag.columns:
            raise ValueError("PatientDiagnosis.csv must have column 'CODI'")

        diag = diag.rename(columns={"CODI": "patient_id"})
        diag["patient_id"] = diag["patient_id"].astype(str)

        diag["patient_label"] = diag.iloc[:, 1].map(
            lambda x: 0 if str(x).upper().startswith("NEG") else 1
        )

        print(f"[INFO] patients in CSV: {diag['patient_id'].nunique()}")

    # -------------------------
    # split
    # -------------------------
    if diag is None:
        train_df, val_df, test_df = patch_level_split(df, seed=args.seed)
    else:
        df["patient_id"] = df["Pat_ID"].astype(str)
        patients = df["patient_id"].unique()

        rng = np.random.RandomState(args.seed)
        rng.shuffle(patients)

        n = len(patients)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        train_p = set(patients[:n_train])
        val_p = set(patients[n_train : n_train + n_val])
        test_p = set(patients[n_train + n_val :])

        train_df = df[df["patient_id"].isin(train_p)]
        val_df = df[df["patient_id"].isin(val_p)]
        test_df = df[df["patient_id"].isin(test_p)]

    # -------------------------
    # save
    # -------------------------
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("[INFO] split done")
    print(f"  train: {len(train_df)}")
    print(f"  val  : {len(val_df)}")
    print(f"  test : {len(test_df)}")


if __name__ == "__main__":
    main()
