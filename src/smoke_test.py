import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # ImageNet normalization (good default for ResNet pretrained)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_image(path: str, tfm):
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0)  # [1, 3, H, W]
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, required=True, help="Path to image 1")
    parser.add_argument("--img2", type=str, default=None, help="Optional path to image 2")
    parser.add_argument("--label1", type=int, default=0, choices=[0, 1], help="Label for image 1")
    parser.add_argument("--label2", type=int, default=1, choices=[0, 1], help="Label for image 2 (if provided)")
    parser.add_argument("--img_size", type=int, default=224, help="Resize size for ResNet input")
    parser.add_argument("--outdir", type=str, default="outputs/smoke", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Model: ResNet18 -> binary classifier
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.to(device)
    model.train()

    tfm = build_transform(args.img_size)

    # Collect samples
    samples = [(args.img1, args.label1)]
    if args.img2:
        samples.append((args.img2, args.label2))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    preds_out = []
    for i, (img_path, y) in enumerate(samples, start=1):
        x = load_image(img_path, tfm).to(device)
        y = torch.tensor([y], dtype=torch.long, device=device)

        logits = model(x)                 # [1, 2]
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = torch.softmax(logits.detach().cpu(), dim=1).numpy().tolist()[0]
        pred = int(torch.argmax(logits.detach().cpu(), dim=1).item())

        print(f"[{i}] path={img_path}")
        print(f"    input_shape={tuple(x.shape)} loss={float(loss.item()):.6f}")
        print(f"    probs=[neg={probs[0]:.4f}, pos={probs[1]:.4f}] pred={pred} label={int(y.item())}")

        preds_out.append({
            "path": img_path,
            "label": int(y.item()),
            "pred": pred,
            "prob_neg": probs[0],
            "prob_pos": probs[1],
            "loss": float(loss.item()),
        })

    # Save checkpoint + predictions
    ckpt_path = outdir / "model.ckpt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "img_size": args.img_size,
        "seed": args.seed,
    }, ckpt_path)

    preds_path = outdir / "preds.json"
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(preds_out, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {ckpt_path}")
    print(f"Saved: {preds_path}")


if __name__ == "__main__":
    main()
