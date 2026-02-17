import argparse
from pathlib import Path

import pandas as pd

from metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", type=str, required=True,
                    help="CSV with columns: patient_id,label,prob_pos,pred")
    ap.add_argument("--th", type=float, default=0.5, help="threshold on mean prob_pos")
    ap.add_argument("--out_csv", type=str, default="outputs/patient_preds.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    required = {"patient_id", "label", "prob_pos"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"pred_csv missing columns: {missing}")

    g = df.groupby("patient_id").agg(
        prob_pos_mean=("prob_pos", "mean"),
        label=("label", "max"),
    ).reset_index()

    g["pred"] = (g["prob_pos_mean"] >= args.th).astype(int)

    y_true = g["label"].astype(int).tolist()
    y_pred = g["pred"].astype(int).tolist()
    y_prob = g["prob_pos_mean"].astype(float).tolist()

    m = compute_metrics(y_true, y_pred, y_prob)
    print("[PATIENT] metrics:", m)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(out, index=False)
    print(f"[OK] wrote patient preds: {out}")


if __name__ == "__main__":
    main()
