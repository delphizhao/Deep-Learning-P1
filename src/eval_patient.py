# src/eval_patient.py
import argparse
from pathlib import Path
import os

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet50


# -----------------------------
# Utils
# -----------------------------
def pick_col(df: pd.DataFrame, candidates):
    """Return the first existing column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def sigmoid_or_softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    If logits has shape (N,2) -> softmax and return prob of class 1.
    If logits has shape (N,) or (N,1) -> sigmoid.
    """
    if logits.ndim == 2 and logits.size(1) == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    if logits.ndim == 2 and logits.size(1) == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


def safe_auc(y_true: np.ndarray, y_score: np.ndarray):
    # Optional dependency (nice to have)
    try:
        from sklearn.metrics import roc_auc_score
        # roc_auc_score fails if only one class present
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def confusion(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


# -----------------------------
# Dataset
# -----------------------------
class PatchCSV(Dataset):
    """
    Reads a CSV with at least:
      - path OR (img_root + rel_path)
      - label (optional for inference)
      - Pat_ID / patient_id (for patient aggregation)

    Your current prepare_splits.py writes: ... label, rel_path, path
    """
    def __init__(self, csv_path: str, img_root: str = None, transform=None):
        self.csv_path = str(csv_path)
        self.img_root = str(img_root) if img_root is not None else None
        self.transform = transform

        df = pd.read_csv(self.csv_path)

        # columns
        self.col_path = pick_col(df, ["path", "img_path", "image_path"])
        self.col_rel = pick_col(df, ["rel_path", "relative_path"])
        self.col_label = pick_col(df, ["label", "Label", "y"])
        self.col_pid = pick_col(df, ["Pat_ID", "patient_id", "CODI", "patient"])

        if self.col_pid is None:
            raise ValueError(f"Cannot find patient id column in {self.csv_path}. "
                             f"Columns={list(df.columns)}")

        if self.col_path is None and self.col_rel is None:
            raise ValueError(f"CSV must contain 'path' or 'rel_path'. Columns={list(df.columns)}")

        # build final path list
        paths = []
        for _, row in df.iterrows():
            if self.col_path is not None and isinstance(row[self.col_path], str) and len(row[self.col_path]) > 0:
                p = row[self.col_path]
            else:
                if self.img_root is None:
                    raise ValueError("img_root is required when CSV has only rel_path")
                p = str(Path(self.img_root) / str(row[self.col_rel]))
            paths.append(p)

        df = df.copy()
        df["_final_path"] = paths
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["_final_path"]
        pid = row[self.col_pid]

        # label might not exist (inference-only)
        label = -1
        if self.col_label is not None:
            try:
                label = int(row[self.col_label])
            except Exception:
                label = -1

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # prevent crash if a file is corrupted/missing
            img = Image.new("RGB", (224, 224))

        if self.transform:
            img = self.transform(img)

        return img, label, str(pid), img_path


# -----------------------------
# Model loading
# -----------------------------
def build_model(name: str):
    name = name.lower()
    if name == "resnet18":
        m = resnet18(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 2)
        return m
    if name == "resnet50":
        m = resnet50(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 2)
        return m
    raise ValueError("model must be resnet18 or resnet50")


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # common patterns
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            # might already be a state dict-like
            state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
            if len(state) == 0:
                state = ckpt
    else:
        state = ckpt

    # handle DataParallel "module."
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    return missing, unexpected


# -----------------------------
# Patient aggregation
# -----------------------------
def aggregate_patient(df_patch: pd.DataFrame, method: str = "mean", topk: int = 10):
    """
    df_patch must contain columns: patient_id, prob
    returns df_patient: patient_id, score
    """
    method = method.lower()

    if method == "mean":
        g = df_patch.groupby("patient_id")["prob"].mean()
        return g.reset_index().rename(columns={"prob": "score"})

    if method == "max":
        g = df_patch.groupby("patient_id")["prob"].max()
        return g.reset_index().rename(columns={"prob": "score"})

    if method == "vote":
        # majority vote on prob>=0.5
        pred_patch = (df_patch["prob"].values >= 0.5).astype(int)
        tmp = df_patch.copy()
        tmp["pred_patch"] = pred_patch
        g = tmp.groupby("patient_id")["pred_patch"].mean()  # fraction of positive votes
        return g.reset_index().rename(columns={"pred_patch": "score"})

    if method == "topk_mean":
        # mean of top-k probabilities per patient
        def topk_mean(x):
            xs = np.sort(x.values)[::-1]
            k = min(topk, len(xs))
            return float(xs[:k].mean())
        g = df_patch.groupby("patient_id")["prob"].apply(topk_mean)
        return g.reset_index().rename(columns={"prob": "score"})

    raise ValueError("method must be one of: mean, max, vote, topk_mean")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True, help="e.g. splits/test.csv")
    parser.add_argument("--ckpt", type=str, required=True, help="e.g. runs/resnet50_seed42/best.pth")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--img_root", type=str, default=None, help="Only needed if test_csv has rel_path but no path")
    parser.add_argument("--patient_csv", type=str, default=None, help="Optional PatientDiagnosis.csv for ground truth")
    parser.add_argument("--patient_id_col", type=str, default=None, help="Optional override, e.g. CODI")
    parser.add_argument("--patient_label_col", type=str, default=None, help="Optional override label column name")
    parser.add_argument("--agg", type=str, default="mean", choices=["mean", "max", "vote", "topk_mean"])
    parser.add_argument("--topk", type=int, default=10, help="Used when agg=topk_mean")
    parser.add_argument("--threshold", type=float, default=0.5, help="Patient-level threshold on score")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="runs/patient_eval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # IMPORTANT: match training normalization (ImageNet norm is typical for ResNet)
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds = PatchCSV(args.test_csv, img_root=args.img_root, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = build_model(args.model).to(device)
    model.eval()
    missing, unexpected = load_checkpoint(model, args.ckpt, device)
    print(f"[INFO] loaded ckpt = {args.ckpt}")
    if missing:
        print(f"[WARN] missing keys (show up to 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] unexpected keys (show up to 10): {unexpected[:10]}")

    # 1) Patch-level inference
    rows = []
    with torch.no_grad():
        for imgs, labels, pids, paths in dl:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs = sigmoid_or_softmax_probs(logits).detach().cpu().numpy()

            labels_np = np.array(labels)
            for prob, y, pid, p in zip(probs, labels_np, pids, paths):
                rows.append({
                    "patient_id": str(pid),
                    "path": str(p),
                    "label_patch": int(y),
                    "prob": float(prob),
                    "pred_patch": int(prob >= 0.5),
                })

    df_patch = pd.DataFrame(rows)
    patch_out = out_dir / "patch_preds.csv"
    df_patch.to_csv(patch_out, index=False)
    print(f"[INFO] wrote {patch_out} (n={len(df_patch)})")

    # 2) Patient-level aggregation
    df_patient = aggregate_patient(df_patch, method=args.agg, topk=args.topk)
    df_patient["pred_patient"] = (df_patient["score"].values >= args.threshold).astype(int)

    # 3) Optional: merge patient-level ground truth
    if args.patient_csv is not None and Path(args.patient_csv).exists():
        diag = pd.read_csv(args.patient_csv)

        pid_col = args.patient_id_col or pick_col(diag, ["patient_id", "CODI", "Pat_ID", "patient"])
        if pid_col is None:
            raise ValueError(f"Cannot find patient id column in patient_csv. Columns={list(diag.columns)}")

        # label col: if not provided, use 2nd column heuristic OR common names
        label_col = args.patient_label_col or pick_col(diag, ["patient_label", "label", "Diagnosis", "diagnosis"])
        if label_col is None:
            # fallback: second column
            if diag.shape[1] >= 2:
                label_col = diag.columns[1]
            else:
                raise ValueError("patient_csv has no label-like column.")

        # map label to 0/1 (robust)
        diag = diag.copy()
        diag["patient_id"] = diag[pid_col].astype(str)

        def map_label(x):
            s = str(x).strip().upper()
            # common patterns: NEG / POS / 0 / 1
            if s.startswith("NEG") or s in ["0", "NO", "FALSE"]:
                return 0
            if s.startswith("POS") or s in ["1", "YES", "TRUE"]:
                return 1
            # fallback: try numeric
            try:
                v = float(x)
                return 1 if v > 0 else 0
            except Exception:
                return np.nan

        diag["label_patient"] = diag[label_col].apply(map_label)
        diag = diag.dropna(subset=["label_patient"])
        diag["label_patient"] = diag["label_patient"].astype(int)

        df_patient = df_patient.merge(
            diag[["patient_id", "label_patient"]],
            on="patient_id",
            how="left"
        )

        # metrics
        valid = df_patient.dropna(subset=["label_patient"]).copy()
        if len(valid) == 0:
            print("[WARN] No patient labels matched. Check patient_id mapping.")
        else:
            y_true = valid["label_patient"].values.astype(int)
            y_pred = valid["pred_patient"].values.astype(int)
            y_score = valid["score"].values.astype(float)

            acc = float((y_true == y_pred).mean())
            tp, tn, fp, fn = confusion(y_true, y_pred)
            auc = safe_auc(y_true, y_score)

            print("\n[RESULT] Patient-level evaluation")
            print(f"  agg      = {args.agg} (topk={args.topk} if used)")
            print(f"  thr      = {args.threshold}")
            print(f"  patients = {len(valid)}")
            print(f"  acc      = {acc:.4f}")
            print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
            if auc is not None:
                print(f"  AUC      = {auc:.4f}")
            else:
                print("  AUC      = N/A (need both classes + sklearn)")

    patient_out = out_dir / "patient_preds.csv"
    df_patient.to_csv(patient_out, index=False)
    print(f"[INFO] wrote {patient_out} (n={len(df_patient)})")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
