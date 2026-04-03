"""
Adapt the 2-class trained checkpoint to a 4-class segmentation model.

Loads the pre-trained encoder weights from the binary segmentation model
and fine-tunes a new 4-class head. Supports both real BraTS data and
synthetic data (synthetic is used only when no real data is available).

Usage:
  # With real BraTS data (recommended):
  python adapt_model.py --data_dir /path/to/BraTS

  # With synthetic data (fallback for demo/testing):
  python adapt_model.py --synthetic
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).parent))

from model import MultiPathFusionNet, CombinedLoss, compute_all_dice
from train import SyntheticBraTSDataset, BraTSDataset


def adapt_checkpoint(
    src_path: str = "checkpoints/best_model_2class.pth",
    dst_path: str = "checkpoints/best_model.pth",
    data_dir: str = None,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 5e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load 2-class checkpoint ─────────────────────────────
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    sd_2class = ckpt["model_state"]
    dice_info = ckpt.get('tumor_dice', ckpt.get('dice_scores', 'N/A'))
    print(f"Loaded 2-class checkpoint (epoch {ckpt['epoch']}, dice {dice_info})")

    # ── Create 4-class model and load encoder weights ───────
    model = MultiPathFusionNet(in_channels=1, num_classes=4).to(device)

    # Remove head weights (shape mismatch: 2 vs 4 classes)
    encoder_sd = {k: v for k, v in sd_2class.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(encoder_sd, strict=False)
    print(f"Loaded {len(encoder_sd)}/{len(sd_2class)} parameters (missing: {missing})")

    # Freeze encoder, train only the head initially
    for name, param in model.named_parameters():
        if not name.startswith("head."):
            param.requires_grad = False

    # ── Datasets ────────────────────────────────────────────
    if data_dir:
        print(f"Using REAL BraTS data from: {data_dir}")
        train_ds = BraTSDataset(data_dir, img_size=256, in_channels=1, split="train")
        val_ds = BraTSDataset(data_dir, img_size=256, in_channels=1, split="val")
    else:
        print("Using SYNTHETIC data for adaptation (use --data_dir for real BraTS data)")
        train_ds = SyntheticBraTSDataset(n_samples=100, img_size=256)
        val_ds = SyntheticBraTSDataset(n_samples=20, img_size=256)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Phase 1: Train head only ────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    criterion = CombinedLoss(alpha=0.5)

    print(f"\n── Phase 1: Training head only ({epochs // 2} epochs) ──")
    best_dice = 0.0
    best_state = None

    for epoch in range(1, epochs // 2 + 1):
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

        avg_dice = _validate(model, val_loader, device)
        print(f"  Epoch {epoch:2d} | Loss: {total_loss / len(train_loader):.4f} | Avg Dice: {avg_dice:.1f}%")

        if avg_dice > best_dice:
            best_dice = avg_dice
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Phase 2: Fine-tune all layers ───────────────────────
    print(f"\n── Phase 2: Fine-tuning all layers ({epochs - epochs // 2} epochs) ──")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - epochs // 2)

    for epoch in range(epochs // 2 + 1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_dice = _validate(model, val_loader, device)
        print(f"  Epoch {epoch:2d} | Loss: {total_loss / len(train_loader):.4f} | Avg Dice: {avg_dice:.1f}%")

        if avg_dice > best_dice:
            best_dice = avg_dice
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Save adapted checkpoint ─────────────────────────────
    model.load_state_dict(best_state)
    torch.save({
        "epoch": epochs,
        "model_state": best_state,
        "dice_scores": _validate_detailed(model, val_loader, device),
        "adapted_from": "best_model_2class.pth",
        "num_classes": 4,
    }, dst_path)

    print(f"\n✓ Saved 4-class model → {dst_path} (Best Avg Dice: {best_dice:.1f}%)")
    return dst_path


def _validate(model, val_loader, device):
    model.eval()
    total_dice = 0.0
    n = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            pred = model(imgs).argmax(dim=1).cpu()
            for i in range(imgs.size(0)):
                d = compute_all_dice(pred[i], masks[i])
                total_dice += sum(d.values()) / 3
                n += 1
    return total_dice / n if n > 0 else 0.0


def _validate_detailed(model, val_loader, device):
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
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapt 2-class model to 4-class")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to real BraTS dataset (NIfTI files)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (default if no --data_dir)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    adapt_checkpoint(data_dir=args.data_dir, epochs=args.epochs,
                     batch_size=args.batch_size, lr=args.lr)
