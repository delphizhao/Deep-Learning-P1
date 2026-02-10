import pandas as pd
from pathlib import Path

# 1) read patch annotations
xlsx_path = Path("HP_WSI-CoordAnnotatedPatches.xlsx")
df = pd.read_excel(xlsx_path)

# keep only valid labels
df = df[df["Presence"].isin([1, -1])].copy()
df["label"] = (df["Presence"] == 1).astype(int)

# build path: Pat_ID_SectionID/Window_ID.png
# folder name described as PatID_Section# (Section_ID is 0/1) :contentReference[oaicite:12]{index=12}
def make_rel_path(row):
    folder = f"{row['Pat_ID']}_{int(row['Section_ID'])}"
    fname = f"{int(row['Window_ID'])}.png"
    return str(Path(folder) / fname)

df["rel_path"] = df.apply(make_rel_path, axis=1)
df["patient_id"] = df["Pat_ID"]
df["section_id"] = df["Section_ID"]
df["window_id"] = df["Window_ID"]
df["source"] = "annotated_excel"

# 2) read patient diagnosis (for patient-level split / stratification)
diag = pd.read_csv("PatientDiagnosis.csv")
# columns: CODI, diagnosis (NEGATIVA/BAIXA/ALTA) :contentReference[oaicite:13]{index=13}
diag = diag.rename(columns={"CODI": "patient_id"})
diag["patient_label"] = diag.iloc[:, 1].map(lambda x: 0 if str(x).upper().startswith("NEG") else 1)

# Later: split on diag.patient_id, then filter df by patient_id
