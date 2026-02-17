import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset_csv import CSVPatchDataset
from metrics import compute_metrics


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def make_model(name: str):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError("model must be resnet18 or resnet50")

    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, 2)
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    rows = []

    for x, y, patient_id in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
        pred = torch.argmax(logits, dim=1).detach().cpu().tolist()
        yt = y.detach().cpu().tolist()

        y_true.extend(yt)
        y_pred.extend(pred)
        y_prob.extend(prob)

        for pid, t, pprob, ppred in zip(patient_id, yt, prob, pred):
            rows.append({"patient_id": pid, "label": int(t), "prob_pos": float(pprob), "pred": int(ppred)})

    metrics = compute_metrics(y_true, y_pred, y_prob)
    return metrics, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="outputs/cnn")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CSVPatchDataset(args.train_csv, transform=build_transforms(train=True))
    val_ds = CSVPatchDataset(args.val_csv, transform=build_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # class weights (handle imbalance)
    labels = train_ds.df["label"].astype(int).tolist()
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    w_pos = n_neg / max(n_pos, 1)
    weight = torch.tensor([1.0, float(w_pos)], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    model = make_model(args.model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_auc = -1.0
    best_path = out_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for x, y, _pid in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        train_loss = running / max(len(train_loader), 1)

        val_metrics, val_rows = evaluate(model, val_loader, device)
        print(f"[E{epoch}] train_loss={train_loss:.4f} val={val_metrics}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), best_path)
            (out_dir / "patch_preds_val.csv").write_text(
                "\n".join(["patient_id,label,prob_pos,pred"] + [
                    f"{r['patient_id']},{r['label']},{r['prob_pos']},{r['pred']}" for r in val_rows
                ]),
                encoding="utf-8",
            )
            with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump({"epoch": epoch, "train_loss": train_loss, **val_metrics}, f, indent=2)
            print(f"[OK] saved best -> {best_path} (AUC={best_auc:.4f})")

    print("[DONE] CNN training finished.")
    print(f"Best model: {best_path}")
    print(f"Val patch preds: {out_dir / 'patch_preds_val.csv'}")


if __name__ == "__main__":
    main()
