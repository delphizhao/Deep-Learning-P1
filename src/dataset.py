# src/dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class HPExcelDataset(Dataset):
    """
    Read the teacher-provided Excel (AnnotatedAllPatches / AnnotatedPatches),
    filter Presence in {1, -1}, map label: 1 -> 1, -1 -> 0
    and load images from:
        img_root / f"{Pat_ID}_{Section_ID}" / f"{Window_ID}.png"
    """

    def __init__(
        self,
        excel_path: str | Path,
        img_root: str | Path,
        transform=None,
        strict: bool = False,
    ):
        self.excel_path = Path(excel_path)
        self.img_root = Path(img_root)
        self.transform = transform
        self.strict = strict

        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel not found: {self.excel_path}")
        if not self.img_root.exists():
            raise FileNotFoundError(f"img_root not found: {self.img_root}")

        df = pd.read_excel(self.excel_path)

        if "Presence" not in df.columns:
            raise ValueError("Excel missing column: Presence")

        df = df[df["Presence"].isin([1, -1])].copy()
        df["label"] = (df["Presence"] == 1).astype(int)

        # Some files may have Window_ID like '902_Aug1' -> DO NOT cast to int blindly
        # Build filename as string + '.png'
        def make_abs_path(row):
            folder = f"{row['Pat_ID']}_{row['Section_ID']}"
            fname = f"{str(row['Window_ID'])}.png"
            return str(self.img_root / folder / fname)

        df["_abs_path"] = df.apply(make_abs_path, axis=1)

        exists_mask = df["_abs_path"].apply(lambda p: Path(p).exists())
        missing = int((~exists_mask).sum())
        if missing > 0:
            print(f"[dataset] Missing files: {missing} / {len(df)} (will {'error' if strict else 'drop'})")
            if strict:
                ex = df.loc[~exists_mask, "_abs_path"].head(10).tolist()
                raise FileNotFoundError("Example missing paths:\n" + "\n".join(ex))
            df = df.loc[exists_mask].reset_index(drop=True)

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["_abs_path"]
        label = int(row["label"])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            if self.strict:
                raise
            img = Image.new("RGB", (224, 224))

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
