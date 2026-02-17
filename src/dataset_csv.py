# src/dataset_csv.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class CSVSpec:
    path_col: str = "path"          # absolute path preferred
    rel_path_col: str = "rel_path"  # fallback if "path" not present
    label_col: str = "label"


class HPTileCSVDataset(Dataset):
    """
    Dataset that reads a split CSV and loads images.

    CSV is expected to have:
      - label column: default "label" (0/1)
      - either:
          * "path" column with absolute image path, OR
          * "rel_path" column, plus you pass img_root to build absolute path
    """

    def __init__(
        self,
        csv_path: str | Path,
        img_root: Optional[str | Path] = None,
        transform=None,
        strict: bool = False,
        spec: CSVSpec = CSVSpec(),
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.img_root = Path(img_root) if img_root is not None else None
        self.transform = transform
        self.strict = strict
        self.spec = spec

        df = pd.read_csv(self.csv_path)

        if self.spec.label_col not in df.columns:
            raise ValueError(
                f"CSV missing label column '{self.spec.label_col}'. "
                f"Columns={list(df.columns)}"
            )

        # Resolve image path column
        if self.spec.path_col in df.columns:
            df["_abs_path"] = df[self.spec.path_col].astype(str)
        elif self.spec.rel_path_col in df.columns:
            if self.img_root is None:
                raise ValueError(
                    f"CSV has '{self.spec.rel_path_col}' but no '{self.spec.path_col}'. "
                    f"Please pass --img_root to build absolute paths."
                )
            df["_abs_path"] = df[self.spec.rel_path_col].astype(str).apply(
                lambda p: str(self.img_root / p)
            )
        else:
            raise ValueError(
                f"CSV missing both '{self.spec.path_col}' and '{self.spec.rel_path_col}'. "
                f"Columns={list(df.columns)}"
            )

        # Clean labels -> int 0/1
        df["_label"] = df[self.spec.label_col].astype(int)

        # Optional: filter missing files (recommended)
        exists_mask = df["_abs_path"].apply(lambda p: Path(p).exists())
        missing = int((~exists_mask).sum())
        if missing > 0:
            msg = f"[dataset_csv] Missing files: {missing} / {len(df)} (will {'error' if strict else 'drop'})"
            print(msg)
            if strict:
                # show up to 10 missing samples
                ex = df.loc[~exists_mask, "_abs_path"].head(10).tolist()
                raise FileNotFoundError("Example missing paths:\n" + "\n".join(ex))
            df = df.loc[exists_mask].reset_index(drop=True)

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            if self.strict:
                raise
            # fallback: black image to keep training running (but you should avoid this on final run)
            # 224 is typical for resnet
            return Image.new("RGB", (224, 224))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["_abs_path"]
        label = int(row["_label"])

        img = self._load_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
