# src/group_split.py
# Group split by patient (or WSI) to avoid leakage.
# Output: train/val/test patch-level CSVs where each patient appears in only one split.

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd


REQUIRED_PATCH_COLS = ["Pat_ID"]
# We will try to find an image path column from these candidates:
PATH_CANDIDATES = ["path", "img_path", "image_path", "filepath", "file_path"]
REL_PATH_CANDIDATES = ["rel_path", "relative_path"]


def _pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_out_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _normalize_label_series(s: pd.Series) -> pd.Series:
    """
    Normalize labels to {0,1}. Accepts:
      - {0,1}
      - {-1,0,1} where -1 means negative (0), 1 means positive (1), 0 maybe unknown -> treated as 0 by default
      - strings like '0','1','-1'
    """
    s2 = pd.to_numeric(s, errors="coerce")
    # If there are -1/1 labels, map -1 -> 0, 1 -> 1
    # If there are 0/1 labels, keep them
    # If there are NaNs, keep NaN (filtered later if needed)
    s2 = s2.replace({-1: 0})
    # Clip any weird values to 0/1 if possible
    s2 = s2.where(s2.isna(), s2.astype(int))
    s2 = s2.where(s2.isna(), s2.clip(lower=0, upper=1))
    return s2


def _derive_patient_labels_from_patches(
    patch_df: pd.DataFrame,
    patient_col: str,
    patch_label_col: str,
    agg: str = "max",
) -> pd.DataFrame:
    """
    Build patient-level labels from patch labels.
    agg='max' is common for MIL: if any patch is positive => patient positive.
    agg='mean' would be probability-like but needs threshold later; here we output hard label.
    """
    tmp = patch_df[[patient_col, patch_label_col]].copy()
    tmp[patch_label_col] = _normalize_label_series(tmp[patch_label_col])
    tmp = tmp.dropna(subset=[patch_label_col])

    if agg == "max":
        pl = tmp.groupby(patient_col)[patch_label_col].max().reset_index()
    elif agg == "mean":
        pl = tmp.groupby(patient_col)[patch_label_col].mean().reset_index()
        pl[patch_label_col] = (pl[patch_label_col] >= 0.5).astype(int)
    else:
        raise ValueError(f"Unsupported agg={agg}. Use max or mean.")

    pl = pl.rename(columns={patch_label_col: "patient_label"})
    return pl


def _load_patient_labels(
    patient_csv: str,
    patient_col: str,
    patient_label_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(patient_csv)
    if patient_col not in df.columns:
        raise ValueError(f"patient_csv missing patient_col='{patient_col}'. Columns: {list(df.columns)}")
    if patient_label_col not in df.columns:
        raise ValueError(f"patient_csv missing patient_label_col='{patient_label_col}'. Columns: {list(df.columns)}")

    out = df[[patient_col, patient_label_col]].copy()
    out[patient_label_col] = _normalize_label_series(out[patient_label_col])
    out = out.dropna(subset=[patient_label_col])
    out = out.rename(columns={patient_label_col: "patient_label"})
    out["patient_label"] = out["patient_label"].astype(int)
    return out


def _stratified_split_patients(
    patient_labels: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split by patient_label (0/1). Returns arrays of patient IDs for train/val/test.
    No sklearn dependency.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "ratios must sum to 1"

    rng = np.random.default_rng(seed)

    pos = patient_labels[patient_labels["patient_label"] == 1]["Pat_ID"].to_numpy()
    neg = patient_labels[patient_labels["patient_label"] == 0]["Pat_ID"].to_numpy()

    rng.shuffle(pos)
    rng.shuffle(neg)

    def split_one(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(arr)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # ensure not exceed
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val
        tr = arr[:n_train]
        va = arr[n_train:n_train + n_val]
        te = arr[n_train + n_val:]
        assert len(te) == n_test
        return tr, va, te

    tr_p, va_p, te_p = split_one(pos)
    tr_n, va_n, te_n = split_one(neg)

    train_ids = np.concatenate([tr_p, tr_n])
    val_ids = np.concatenate([va_p, va_n])
    test_ids = np.concatenate([te_p, te_n])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    return train_ids, val_ids, test_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_csv", required=True, help="Patch-level CSV containing Pat_ID and image path columns.")
    ap.add_argument("--patient_csv", default=None, help="Optional patient-level CSV (Pat_ID + label).")
    ap.add_argument("--patient_col", default="Pat_ID", help="Patient ID column name.")
    ap.add_argument("--patch_label_col", default="label", help="Patch label column name in patch_csv.")
    ap.add_argument("--patient_label_col", default="label", help="Patient label column name in patient_csv.")
    ap.add_argument("--agg", default="max", choices=["max", "mean"], help="How to derive patient label from patch labels if patient_csv not provided.")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="splits_group")
    ap.add_argument("--require_existing_files", action="store_true",
                    help="If set, drop rows whose image path does not exist on disk (slow if huge).")
    args = ap.parse_args()

    out_dir = _ensure_out_dir(args.out_dir)

    patch_df = pd.read_csv(args.patch_csv)

    # Basic checks
    if args.patient_col not in patch_df.columns:
        raise ValueError(f"patch_csv missing patient_col='{args.patient_col}'. Columns: {list(patch_df.columns)}")

    # Determine path column
    path_col = _pick_first_existing_col(patch_df, PATH_CANDIDATES)
    rel_path_col = _pick_first_existing_col(patch_df, REL_PATH_CANDIDATES)

    if path_col is None and rel_path_col is None:
        raise ValueError(
            f"patch_csv must contain an image path column. Tried {PATH_CANDIDATES + REL_PATH_CANDIDATES}. "
            f"Columns: {list(patch_df.columns)}"
        )

    # If patch label col exists, normalize it (used for patient label derivation)
    if args.patch_label_col in patch_df.columns:
        patch_df[args.patch_label_col] = _normalize_label_series(patch_df[args.patch_label_col])
    else:
        # If no patch label, we can still split by patient IDs but cannot stratify.
        # We'll create dummy label=0 and do non-stratified split.
        patch_df[args.patch_label_col] = np.nan

    # Optionally filter missing files
    if args.require_existing_files:
        col = path_col if path_col is not None else rel_path_col
        exists_mask = patch_df[col].apply(lambda p: isinstance(p, str) and os.path.exists(p))
        before = len(patch_df)
        patch_df = patch_df[exists_mask].reset_index(drop=True)
        print(f"[INFO] filtered non-existing files: {before} -> {len(patch_df)}")

    # Build patient_labels table
    if args.patient_csv is not None:
        patient_labels = _load_patient_labels(args.patient_csv, args.patient_col, args.patient_label_col)
        patient_labels = patient_labels.rename(columns={args.patient_col: "Pat_ID"})
    else:
        # Derive from patch labels
        if args.patch_label_col not in patch_df.columns:
            raise ValueError("No patch_label_col in patch_csv and no patient_csv provided; cannot build patient labels.")
        derived = _derive_patient_labels_from_patches(
            patch_df.rename(columns={args.patient_col: "Pat_ID"}),
            patient_col="Pat_ID",
            patch_label_col=args.patch_label_col,
            agg=args.agg,
        )
        patient_labels = derived

    # Keep only patients that appear in patch_df
    patch_patients = set(patch_df[args.patient_col].astype(str).unique())
    patient_labels["Pat_ID"] = patient_labels["Pat_ID"].astype(str)
    patient_labels = patient_labels[patient_labels["Pat_ID"].isin(patch_patients)].reset_index(drop=True)

    if len(patient_labels) == 0:
        raise RuntimeError("No patient labels matched patch_csv patients. Check Pat_ID format / CSVs.")

    # If labels are missing / not binary, fallback to non-stratified split
    unique_labels = sorted(patient_labels["patient_label"].dropna().unique().tolist())
    can_stratify = (set(unique_labels).issubset({0, 1}) and len(unique_labels) >= 1)

    rng = np.random.default_rng(args.seed)

    if can_stratify and len(unique_labels) == 2:
        train_ids, val_ids, test_ids = _stratified_split_patients(
            patient_labels[["Pat_ID", "patient_label"]],
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
    else:
        # non-stratified
        pats = patient_labels["Pat_ID"].to_numpy()
        rng.shuffle(pats)
        n = len(pats)
        n_train = int(round(n * args.train_ratio))
        n_val = int(round(n * args.val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train_ids = pats[:n_train]
        val_ids = pats[n_train:n_train + n_val]
        test_ids = pats[n_train + n_val:]

    # Map patient split back to patch rows
    patch_df[args.patient_col] = patch_df[args.patient_col].astype(str)

    train_df = patch_df[patch_df[args.patient_col].isin(set(train_ids))].copy()
    val_df = patch_df[patch_df[args.patient_col].isin(set(val_ids))].copy()
    test_df = patch_df[patch_df[args.patient_col].isin(set(test_ids))].copy()

    # Sanity check: no leakage
    tr_p = set(train_df[args.patient_col].unique())
    va_p = set(val_df[args.patient_col].unique())
    te_p = set(test_df[args.patient_col].unique())
    assert len(tr_p & va_p) == 0 and len(tr_p & te_p) == 0 and len(va_p & te_p) == 0, "Leakage detected!"

    # Save
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Print summary
    print("[INFO] group split done (by patient).")
    print(f"[INFO] patients: train={len(tr_p)} val={len(va_p)} test={len(te_p)} total={len(tr_p|va_p|te_p)}")
    print(f"[INFO] patches : train={len(train_df)} val={len(val_df)} test={len(test_df)} total={len(patch_df)}")
    print(f"[INFO] wrote: {train_path}, {val_path}, {test_path}")


if __name__ == "__main__":
    main()
