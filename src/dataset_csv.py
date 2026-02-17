from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Row:
    path: str
    label: int
    patient_id: str


class CSVPatchDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        required = {"path", "label", "patient_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        # ensure types
        df["path"] = df["path"].astype(str)
        df["label"] = df["label"].astype(int)
        df["patient_id"] = df["patient_id"].astype(str)
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        r = self.df.iloc[idx]
        img_path = r["path"]
        label = int(r["label"])
        patient_id = str(r["patient_id"])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long), patient_id
