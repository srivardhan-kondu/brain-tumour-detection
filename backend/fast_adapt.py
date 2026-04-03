"""
Fast adaptation: train ONLY the 4-class head on top of frozen 2-class encoder.
Completes in ~2–5 minutes on CPU.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from model import MultiPathFusionNet, CombinedLoss, compute_all_dice
from train import SyntheticBraTSDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    src = Path("checkpoints/best_model_2class.pth")
    dst = Path("checkpoints/best_model.pth")

    # Load 2-class checkpoint
    ckpt = torch.load(str(src), map_location="cpu", weights_only=False)
    sd = ckpt["model_state"]
    print(f"Loaded 2-class checkpoint (epoch {ckpt['epoch']})")

    # Create 4-class model, load encoder weights (skip head)
    model = MultiPathFusionNet(in_channels=1, num_classes=4).to(device)
    encoder_sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    model.load_state_dict(encoder_sd, strict=False)
    print(f"Loaded {len(encoder_sd)} encoder params, head initialized randomly")

    # Freeze everything except head
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("head.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Training {trainable} / {total} parameters (head only)")

    # Small dataset — enough to learn the 4-class mapping
    train_ds = SyntheticBraTSDataset(n_samples=50, img_size=128)
    val_ds = SyntheticBraTSDataset(n_samples=10, img_size=128)
    train_loader = DataLoader(train_ds, batch_size=25, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=10, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)
    criterion = CombinedLoss(alpha=0.5)

    best_dice = 0.0
    best_state = None

    print(f"\n── Training head only (5 epochs) ──")
    for epoch in range(1, 6):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        dice_sum = 0.0
        n = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                pred = model(imgs).argmax(dim=1).cpu()
                for i in range(imgs.size(0)):
                    d = compute_all_dice(pred[i], masks[i])
                    dice_sum += sum(d.values()) / 3
                    n += 1

        avg_dice = dice_sum / n if n > 0 else 0.0
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:2d} | Loss: {total_loss / len(train_loader):.4f} | Dice: {avg_dice:.1f}% | {elapsed:.1f}s")

        if avg_dice > best_dice:
            best_dice = avg_dice
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Save
    model.load_state_dict(best_state)

    # Validate detailed
    model.eval()
    scores = {"whole_tumor": 0, "tumor_core": 0, "enhancing_tumor": 0}
    n = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            pred = model(imgs).argmax(dim=1).cpu()
            for i in range(imgs.size(0)):
                d = compute_all_dice(pred[i], masks[i])
                for k in scores:
                    scores[k] += d[k]
                n += 1
    for k in scores:
        scores[k] = round(scores[k] / n, 1) if n > 0 else 0.0

    torch.save({
        "epoch": 5,
        "model_state": best_state,
        "dice_scores": scores,
        "adapted_from": "best_model_2class.pth",
        "num_classes": 4,
    }, str(dst))

    print(f"\n✓ Saved 4-class model → {dst}")
    print(f"  Best Avg Dice: {best_dice:.1f}%")
    print(f"  WT: {scores['whole_tumor']}% | TC: {scores['tumor_core']}% | ET: {scores['enhancing_tumor']}%")

if __name__ == "__main__":
    main()
