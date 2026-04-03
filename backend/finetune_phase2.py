"""
Phase 2: Fine-tune ALL layers from the 4-class checkpoint for better TC/ET.
"""
import sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from model import MultiPathFusionNet, CombinedLoss, compute_all_dice
from train import SyntheticBraTSDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    src = Path("checkpoints/best_model.pth")
    ckpt = torch.load(str(src), map_location="cpu", weights_only=False)

    model = MultiPathFusionNet(in_channels=1, num_classes=4).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded 4-class checkpoint (WT:{ckpt['dice_scores']['whole_tumor']}%)")

    # ALL params trainable
    for p in model.parameters():
        p.requires_grad = True

    train_ds = SyntheticBraTSDataset(n_samples=200, img_size=128)
    val_ds = SyntheticBraTSDataset(n_samples=20, img_size=128)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=20, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = CombinedLoss(alpha=0.7)  # Higher Dice weight to push TC/ET

    best_dice = 0.0
    best_state = None

    print(f"\n── Phase 2: Fine-tuning all layers (10 epochs) ──")
    for epoch in range(1, 11):
        t0 = time.time()
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

        model.eval()
        dice_sum = 0.0
        scores = {"whole_tumor": 0, "tumor_core": 0, "enhancing_tumor": 0}
        n = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                pred = model(imgs).argmax(dim=1).cpu()
                for i in range(imgs.size(0)):
                    d = compute_all_dice(pred[i], masks[i])
                    dice_sum += sum(d.values()) / 3
                    for k in scores:
                        scores[k] += d[k]
                    n += 1

        avg_dice = dice_sum / n if n > 0 else 0.0
        for k in scores:
            scores[k] = round(scores[k] / n, 1)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch} | Loss: {total_loss / len(train_loader):.4f} | "
              f"WT: {scores['whole_tumor']}% TC: {scores['tumor_core']}% ET: {scores['enhancing_tumor']}% | {elapsed:.1f}s")

        if avg_dice > best_dice:
            best_dice = avg_dice
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_scores = scores.copy()
            # Save immediately so we don't lose progress if interrupted
            torch.save({
                "epoch": ckpt["epoch"] + epoch,
                "model_state": best_state,
                "dice_scores": best_scores,
                "adapted_from": "best_model_2class.pth",
                "num_classes": 4,
            }, str(src))
            print(f"    → Saved checkpoint (Avg Dice: {best_dice:.1f}%)")

    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_scores = scores

    torch.save({
        "epoch": ckpt["epoch"] + 10,
        "model_state": best_state,
        "dice_scores": best_scores,
        "adapted_from": "best_model_2class.pth",
        "num_classes": 4,
    }, str(src))

    print(f"\n✓ Updated 4-class model → {src}")
    print(f"  Best Avg Dice: {best_dice:.1f}%")
    print(f"  WT: {best_scores['whole_tumor']}% | TC: {best_scores['tumor_core']}% | ET: {best_scores['enhancing_tumor']}%")

if __name__ == "__main__":
    main()
