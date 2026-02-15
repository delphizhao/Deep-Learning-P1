import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Folder containing patient folders like B22-83_1/xxxx.png",
    )
    parser.add_argument("--xlsx", type=str, default="data/meta/HP_WSI-CoordAnnotatedPatches.xlsx")
    parser.add_argument("--patient_csv", type=str, default="data/meta/PatientDiagnosis.csv")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # 1) read patch annotations
    xlsx_path = Path(args.xlsx)
    df = pd.read_excel(xlsx_path)

    # keep only valid labels
    df = df[df["Presence"].isin([1, -1])].copy()
    df["label"] = (df["Presence"] == 1).astype(int)

    # build relative path: Pat_ID_SectionID/Window_ID.png
    def make_rel_path(row):
        folder = f"{row['Pat_ID']}_{int(row['Section_ID'])}"
        fname = f"{int(row['Window_ID'])}.png"
        return str(Path(folder) / fname)

    df["rel_path"] = df.apply(make_rel_path, axis=1)

    # NEW: absolute path column
    df["path"] = df["rel_path"].apply(lambda p: str(data_root / p))

    df["patient_id"] = df["Pat_ID"]
    df["section_id"] = df["Section_ID"]
    df["window_id"] = df["Window_ID"]
    df["source"] = "annotated_excel"

    # quick sanity check
    exists = df["path"].apply(lambda p: Path(p).exists())
    print(f"[INFO] data_root = {data_root}")
    print(f"[INFO] labeled patches (Presence in {{1,-1}}): {len(df)}")
    print(f"[INFO] existing image files: {int(exists.sum())} / {len(exists)}")
    if int(exists.sum()) == 0:
        print("[WARN] 0 files found. data_root is likely wrong (Cropped vs Annotated).")
        print("[WARN] Example expected path:", df["path"].iloc[0])
    else:
        missing = df.loc[~exists, "path"].head(5).tolist()
        if missing:
            print("[INFO] example missing paths:")
            for m in missing:
                print("  -", m)

    # 2) read patient diagnosis (for later split)
    diag = pd.read_csv(Path(args.patient_csv))
    diag = diag.rename(columns={"CODI": "patient_id"})
    diag["patient_label"] = diag.iloc[:, 1].map(lambda x: 0 if str(x).upper().startswith("NEG") else 1)
    print(f"[INFO] patients in PatientDiagnosis.csv: {diag['patient_id'].nunique()}")

    # Next step later: patient-level split + write splits/*.csv


if __name__ == "__main__":
    main()
