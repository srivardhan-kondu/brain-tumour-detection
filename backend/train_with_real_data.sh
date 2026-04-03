#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════════
# ALL-IN-ONE: Download BraTS + Train + Verify
# Run:  cd backend && bash train_with_real_data.sh
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="$SCRIPT_DIR/brats_data"
VENV_DIR="$SCRIPT_DIR/../.venv"

echo "════════════════════════════════════════════════"
echo " Step 1/5: Activate virtual environment"
echo "════════════════════════════════════════════════"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "✓ Activated venv at $VENV_DIR"
else
    echo "✗ No venv found at $VENV_DIR — using system Python"
fi

echo ""
echo "════════════════════════════════════════════════"
echo " Step 2/5: Install dependencies"
echo "════════════════════════════════════════════════"
pip install nibabel kaggle --quiet 2>/dev/null || true
echo "✓ nibabel and kaggle installed"

echo ""
echo "════════════════════════════════════════════════"
echo " Step 3/5: Download BraTS dataset from Kaggle"
echo "════════════════════════════════════════════════"

if [ -d "$DATA_DIR" ] && [ "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "✓ BraTS data already exists at $DATA_DIR — skipping download"
else
    mkdir -p "$DATA_DIR"

    # Check for Kaggle credentials
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo ""
        echo "╔═══════════════════════════════════════════════════════════════╗"
        echo "║  Kaggle API key not found!                                   ║"
        echo "║                                                              ║"
        echo "║  One-time setup (takes 1 minute):                            ║"
        echo "║  1. Go to https://www.kaggle.com/settings                    ║"
        echo "║  2. Scroll to 'API' section → click 'Create New Token'       ║"
        echo "║  3. This downloads kaggle.json                               ║"
        echo "║  4. Run:                                                     ║"
        echo "║     mkdir -p ~/.kaggle                                       ║"
        echo "║     mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json         ║"
        echo "║     chmod 600 ~/.kaggle/kaggle.json                          ║"
        echo "║  5. Re-run this script                                       ║"
        echo "║                                                              ║"
        echo "║  OR: Download manually from Kaggle and extract to:           ║"
        echo "║     $DATA_DIR                                                ║"
        echo "╚═══════════════════════════════════════════════════════════════╝"
        exit 1
    fi

    echo "Downloading BraTS 2020 dataset (this may take a few minutes)..."
    kaggle dataset download -d awsaf49/brats20-dataset-training-validation \
        -p "$DATA_DIR" --unzip

    # If the extracted data is nested, flatten it
    if [ -d "$DATA_DIR/BraTS2020_TrainingData" ]; then
        echo "Reorganizing folder structure..."
        mv "$DATA_DIR/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"/* "$DATA_DIR/" 2>/dev/null || \
        mv "$DATA_DIR/BraTS2020_TrainingData"/* "$DATA_DIR/" 2>/dev/null || true
    fi

    echo "✓ Dataset downloaded and extracted to $DATA_DIR"
fi

# Quick check
N_PATIENTS=$(find "$DATA_DIR" -name "*seg*" -type f | wc -l | tr -d ' ')
echo "  Found $N_PATIENTS patients with segmentation masks"

if [ "$N_PATIENTS" -eq 0 ]; then
    echo "✗ No segmentation files found. Check the data directory structure."
    echo "  Expected: $DATA_DIR/<PatientID>/*_seg.nii.gz"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════"
echo " Step 4/5: Train model on real BraTS data"
echo "════════════════════════════════════════════════"
echo "Training with: $N_PATIENTS patients"
echo "This will take a while on CPU. Use Ctrl+C to stop early (best checkpoint is saved)."
echo ""

python -u train.py \
    --data_dir "$DATA_DIR" \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-3

echo ""
echo "════════════════════════════════════════════════"
echo " Step 5/5: Verify trained model"
echo "════════════════════════════════════════════════"

python -c "
import torch
ckpt = torch.load('checkpoints/best_model.pth', map_location='cpu', weights_only=False)
d = ckpt.get('dice_scores', {})
print(f'Checkpoint epoch: {ckpt.get(\"epoch\", \"?\")}')
print(f'  Whole Tumor:      {d.get(\"whole_tumor\", 0):.1f}%')
print(f'  Tumor Core:       {d.get(\"tumor_core\", 0):.1f}%')
print(f'  Enhancing Tumor:  {d.get(\"enhancing_tumor\", 0):.1f}%')
avg = sum(d.values()) / 3 if d else 0
print(f'  Average Dice:     {avg:.1f}%')
print()
if avg > 80:
    print('✓ Model trained successfully with real BraTS data!')
else:
    print('⚠ Dice scores are low. Consider training for more epochs.')
"

echo ""
echo "════════════════════════════════════════════════"
echo " DONE! Start the server with: python main.py"
echo "════════════════════════════════════════════════"
