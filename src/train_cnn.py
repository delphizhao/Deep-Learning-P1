# src/train_cnn.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset_csv import HPTileCSVDataset


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def build_transforms(img_size: int = 224):
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfm, val_tfm


def build_model(name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unknown model: {name}. Choose from: resnet18, resnet50")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "n": total,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "n": total,
    }


def save_ckpt(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, meta: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "meta": meta,
    }, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default=None)

    # Optional: only needed if your CSV DOES NOT have "path" and only has "rel_path"
    parser.add_argument("--img_root", type=str, default=None)

    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained backbone")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out_dir", type=str, default="runs/train_cnn")
    parser.add_argument("--strict", action="store_true", help="Error if any file missing/unreadable")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    if device.type == "cuda":
        print(f"[INFO] cuda device = {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir) / f"{args.model}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] out_dir = {out_dir}")

    train_tfm, val_tfm = build_transforms(args.img_size)

    train_ds = HPTileCSVDataset(args.train_csv, img_root=args.img_root, transform=train_tfm, strict=args.strict)
    val_ds = HPTileCSVDataset(args.val_csv, img_root=args.img_root, transform=val_tfm, strict=args.strict)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(args.model, num_classes=2, pretrained=args.pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, device, criterion, optimizer)
        va = evaluate(model, val_loader, device, criterion)

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "train_n": tr["n"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "val_n": va["n"],
        }
        history.append(row)

        print(
            f"[E{epoch:02d}] "
            f"train loss={row['train_loss']:.4f} acc={row['train_acc']:.4f} | "
            f"val loss={row['val_loss']:.4f} acc={row['val_acc']:.4f}"
        )

        # save last
        save_ckpt(out_dir / "last.pth", model, optimizer, meta={"args": vars(args), "epoch": epoch})

        # save best
        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            save_ckpt(out_dir / "best.pth", model, optimizer, meta={"args": vars(args), "epoch": epoch})
            print(f"[INFO] best updated: val_acc={best_val_acc:.4f}")

        # write metrics json every epoch
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"best_val_acc": best_val_acc, "history": history}, f, indent=2)

    # optional test
    if args.test_csv is not None:
        test_ds = HPTileCSVDataset(args.test_csv, img_root=args.img_root, transform=val_tfm, strict=args.strict)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        ckpt = torch.load(out_dir / "best.pth", map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        te = evaluate(model, test_loader, device, criterion)
        print(f"[TEST] loss={te['loss']:.4f} acc={te['acc']:.4f} n={te['n']}")
        with open(out_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump(te, f, indent=2)


if __name__ == "__main__":
    main()
