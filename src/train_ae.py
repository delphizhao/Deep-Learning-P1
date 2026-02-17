import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_csv import CSVPatchDataset
from metrics import compute_metrics


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.enc(x)
        xh = self.dec(z)
        return xh


def build_transform():
    # AE prefers [0,1] range; keep it simple
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


@torch.no_grad()
def recon_scores(model, loader, device):
    model.eval()
    scores = []
    labels = []
    patient_ids = []
    for x, y, pid in loader:
        x = x.to(device)
        y = y.to(device)
        xh = model(x)
        # per-sample MSE
        mse = torch.mean((x - xh) ** 2, dim=(1, 2, 3)).detach().cpu().tolist()
        scores.extend([float(s) for s in mse])
        labels.extend(y.detach().cpu().tolist())
        patient_ids.extend(pid)
    return patient_ids, labels, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_neg_csv", type=str, required=True, help="negative-only train split")
    ap.add_argument("--val_csv", type=str, required=True, help="val split (mixed)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="outputs/ae")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = build_transform()
    train_ds = CSVPatchDataset(args.train_neg_csv, transform=tfm)
    val_ds = CSVPatchDataset(args.val_csv, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = ConvAutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_auc = -1.0
    best_path = out_dir / "best_ae.pth"
    best_th = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for x, _y, _pid in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            xh = model(x)
            loss = criterion(xh, x)
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        train_loss = running / max(len(train_loader), 1)

        # validation: scores -> convert to prob_pos via normalization (simple)
        pids, y_true, scores = recon_scores(model, val_loader, device)

        # choose threshold on val by best F1 (simple sweep)
        s_min, s_max = min(scores), max(scores)
        if s_max == s_min:
            s_max = s_min + 1e-6

        # normalize to [0,1] as "prob_pos" (bigger score => more anomalous => more positive)
        prob_pos = [(s - s_min) / (s_max - s_min) for s in scores]

        # sweep thresholds
        best_f1 = -1.0
        th_candidates = [i / 100 for i in range(5, 96, 5)]
        for th in th_candidates:
            y_pred = [1 if p >= th else 0 for p in prob_pos]
            m = compute_metrics(y_true, y_pred, prob_pos)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                th_best = th
                m_best = m

        print(f"[E{epoch}] train_mse={train_loss:.6f} val_best={m_best} th={th_best:.2f}")

        # save by AUC (or F1)
        if m_best["auc"] > best_auc:
            best_auc = m_best["auc"]
            best_th = th_best
            torch.save(model.state_dict(), best_path)
            with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump({"epoch": epoch, "train_mse": train_loss, "threshold": best_th, **m_best}, f, indent=2)

            # write patch preds for patient aggregation
            patch_pred_path = out_dir / "patch_preds_val.csv"
            with open(patch_pred_path, "w", encoding="utf-8") as f:
                f.write("patient_id,label,prob_pos,pred,score\n")
                for pid, yt, pp, sc in zip(pids, y_true, prob_pos, scores):
                    pred = 1 if pp >= best_th else 0
                    f.write(f"{pid},{int(yt)},{float(pp)},{pred},{float(sc)}\n")

            print(f"[OK] saved best AE -> {best_path} (AUC={best_auc:.4f})")

    print("[DONE] AE training finished.")
    print(f"Best AE: {best_path}")
    print(f"Best threshold: {best_th:.2f}")


if __name__ == "__main__":
    main()
