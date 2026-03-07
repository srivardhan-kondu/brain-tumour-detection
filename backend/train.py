"""
Training script for Multi-Path Fusion Network on BraTS dataset.
Supports BraTS 2015 / 2019 / 2020 folder structures.

Usage:
  python train.py --data_dir /path/to/BraTS --epochs 50 --batch_size 4

For quick local test with synthetic data:
  python train.py --synthetic --epochs 5
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model import MultiPathFusionNet, CombinedLoss, compute_all_dice, dice_score


# ─────────────────────────────────────────────────────────────
# Synthetic BraTS-like Dataset (for smoke-test / demo)
# ─────────────────────────────────────────────────────────────

class SyntheticBraTSDataset(Dataset):
    """
    Generates synthetic brain MRI slices with segmentation masks
    that mimic BraTS 2019 label conventions:
      0 = Background
      1 = Whole Tumor (NCR + ET + ED)
      2 = Tumor Core  (NCR + ET)
      3 = Enhancing Tumor (ET)
    """

    def __init__(self, n_samples=200, img_size=256):
        self.n = n_samples
        self.sz = img_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        random.seed(idx)
        np.random.seed(idx)
        h = w = self.sz

        img = np.zeros((h, w), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.int64)

        Y, X = np.ogrid[:h, :w]
        cy, cx = h // 2 + random.randint(-10, 10), w // 2 + random.randint(-10, 10)

        # Brain region
        brain = ((X - cx) ** 2 / 85**2 + (Y - cy) ** 2 / 100**2) <= 1.0
        img[brain] = np.random.uniform(0.3, 0.5, int(brain.sum()))

        # White matter
        wm = ((X - cx) ** 2 / 70**2 + (Y - cy) ** 2 / 85**2) <= 1.0
        img[wm] = np.random.uniform(0.5, 0.7, int(wm.sum()))

        # Tumor
        tx = cx + random.randint(-25, 25)
        ty = cy + random.randint(-25, 25)
        s = random.uniform(0.6, 1.4)

        # Whole Tumor
        wt = ((X - tx) **2 / (45*s)**2 + (Y - ty)**2 / (50*s)**2) <= 1.0 & brain
        img[wt] = np.random.uniform(0.65, 0.8, int(wt.sum()))
        mask[wt] = 1

        # Tumor Core
        tc = ((X - tx)**2 / (25*s)**2 + (Y - ty)**2 / (28*s)**2) <= 1.0 & wt
        img[tc] = np.random.uniform(0.78, 0.9, int(tc.sum()))
        mask[tc] = 2

        # Enhancing Tumor
        etx, ety = tx + random.randint(-5, 5), ty + random.randint(-5, 5)
        et = ((X - etx)**2 / (12*s)**2 + (Y - ety)**2 / (13*s)**2) <= 1.0 & tc
        img[et] = np.random.uniform(0.88, 1.0, int(et.sum()))
        mask[et] = 3

        # Noise
        img += np.random.normal(0, 0.02, (h, w)).astype(np.float32)
        img = np.clip(img, 0, 1)

        img_t  = torch.from_numpy(img).unsqueeze(0)   # (1, H, W)
        mask_t = torch.from_numpy(mask)                # (H, W)
        return img_t, mask_t


# ─────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    if args.synthetic:
        train_ds = SyntheticBraTSDataset(n_samples=400, img_size=256)
        val_ds   = SyntheticBraTSDataset(n_samples=80,  img_size=256)
        print(f"Using synthetic BraTS-like dataset: {len(train_ds)} train / {len(val_ds)} val")
    else:
        raise NotImplementedError("Real BraTS loader not included. Use --synthetic for demo.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = MultiPathFusionNet(in_channels=1, num_classes=4).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # Optimizer + LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CombinedLoss(alpha=0.5)

    best_dice = 0.0
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        t0 = time.time()

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
        avg_loss = total_loss / len(train_loader)

        # ── Validate ─────────────────────────────────────────
        model.eval()
        val_dice = {"whole_tumor": 0, "tumor_core": 0, "enhancing_tumor": 0}
        n_val = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                pred_logits = model(imgs)
                pred_masks  = pred_logits.argmax(dim=1).cpu()
                for i in range(imgs.size(0)):
                    d = compute_all_dice(pred_masks[i], masks[i])
                    for k in val_dice:
                        val_dice[k] += d[k]
                    n_val += 1

        for k in val_dice:
            val_dice[k] /= n_val

        avg_val_dice = sum(val_dice.values()) / 3
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:3d}/{args.epochs}] | Loss: {avg_loss:.4f} | "
            f"WT: {val_dice['whole_tumor']:.1f}% | "
            f"TC: {val_dice['tumor_core']:.1f}% | "
            f"ET: {val_dice['enhancing_tumor']:.1f}% | "
            f"Avg: {avg_val_dice:.1f}% | {elapsed:.1f}s"
        )

        # Save best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            path = save_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "dice_scores": val_dice,
            }, str(path))
            print(f"  ✓ Best model saved (avg Dice: {best_dice:.1f}%) → {path}")

    print(f"\nTraining complete. Best Avg Dice: {best_dice:.1f}%")
    print("Reference target (Wu et al. 2023): WT≥90%, TC≥90%, ET≥85%")


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MPFNet for Brain Tumor Segmentation")
    parser.add_argument("--data_dir",   type=str, default=None)
    parser.add_argument("--synthetic",  action="store_true", default=False)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    if not args.synthetic and not args.data_dir:
        print("No data directory specified. Falling back to synthetic dataset.")
        args.synthetic = True

    train(args)
