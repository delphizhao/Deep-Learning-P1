import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_one(path: str, tfm):
    img = Image.open(path).convert("RGB")
    return tfm(img)  # [3, H, W]


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    # quick mode (2 images)
    p.add_argument("--img1", type=str, default=None)
    p.add_argument("--img2", type=str, default=None)
    p.add_argument("--label1", type=int, default=0, choices=[0, 1])
    p.add_argument("--label2", type=int, default=1, choices=[0, 1])

    # json mode (closer to real pipeline)
    # format: [{"path": "...", "label": 0/1, "patient_id": "P001"}, ...]
    p.add_argument("--items_json", type=str, default=None)

    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--outdir", type=str, default="outputs/smoke")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Build sample list
    items = []
    if args.items_json:
        items = json.loads(Path(args.items_json).read_text(encoding="utf-8"))
    else:
        if not args.img1:
            raise ValueError("Provide --img1/--img2 or --items_json")
        items.append({"path": args.img1, "label": args.label1, "patient_id": "P_demo_1"})
        if args.img2:
            items.append({"path": args.img2, "label": args.label2, "patient_id": "P_demo_2"})

    # Tiny split to mimic your 'train/val/test ratio' idea
    # (for smoke test only; real split will be patient-level)
    train_items = items[: max(1, len(items) - 1)]
    test_items = items[-1:]

    tfm = build_transform(args.img_size)

    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ---- Train (one batch) ----
    model.train()
    xs, ys = [], []
    for it in train_items:
        xs.append(load_one(it["path"], tfm))
        ys.append(it["label"])
    x = torch.stack(xs, dim=0).to(device)                 # [B, 3, H, W]
    y = torch.tensor(ys, dtype=torch.long, device=device)  # [B]

    logits = model(x)                 # [B, 2]
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_probs = torch.softmax(logits.detach().cpu(), dim=1).numpy().tolist()
    train_preds = torch.argmax(logits.detach().cpu(), dim=1).numpy().tolist()

    print(f"[TRAIN] batch_shape={tuple(x.shape)} loss={float(loss.item()):.6f}")
    for i, it in enumerate(train_items):
        print(f"  - {it['path']} label={it['label']} pred={train_preds[i]} "
              f"prob_pos={train_probs[i][1]:.4f}")

    # ---- Eval (test) ----
    model.eval()
    test_out = []
    with torch.no_grad():
        for it in test_items:
            x1 = load_one(it["path"], tfm).unsqueeze(0).to(device)  # [1,3,H,W]
            logit1 = model(x1)
            prob1 = torch.softmax(logit1.cpu(), dim=1).numpy().tolist()[0]
            pred1 = int(torch.argmax(logit1.cpu(), dim=1).item())
            test_out.append({
                "path": it["path"],
                "label": int(it["label"]),
                "pred": pred1,
                "prob_neg": prob1[0],
                "prob_pos": prob1[1],
                "patient_id": it.get("patient_id", ""),
            })
            print(f"[TEST] {it['path']} label={it['label']} pred={pred1} prob_pos={prob1[1]:.4f}")

    # Save artifacts
    torch.save(
        {"model_state_dict": model.state_dict(), "img_size": args.img_size, "seed": args.seed},
        outdir / "model.ckpt",
    )

    metrics = {
        "train_loss": float(loss.item()),
        "num_train": len(train_items),
        "num_test": len(test_items),
        "note": "Smoke test only. Real training uses patient-level split and full dataloader.",
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (outdir / "preds.json").write_text(json.dumps(test_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved: {outdir/'model.ckpt'}")
    print(f"Saved: {outdir/'metrics.json'}")
    print(f"Saved: {outdir/'preds.json'}")


if __name__ == "__main__":
    main()
