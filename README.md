# Brain Tumor Segmentation Dashboard

**Multi-Path Fusion Network with Global Attention**  
Based on: Wu, D., Qiu, S., Qin, J., & Zhao, P. (2023). *Multi-Path Fusion Network Based Global Attention for Brain Tumor Segmentation*. ISAIMS 2023.

**Anurag University — Dept. of CSE**  
Team: Manoj Seemala · Vijay Kumar Anthati · Shaik Ale Shidh Salman · B Sathvika Reddy  
Guide: Mrs. Y. Ashwini, Assistant Professor

---

## Features

| Feature | Details |
|---|---|
| **Model** | Multi-Path Fusion Network + CBAM Global Attention |
| **Input** | 256×256 MRI (DICOM / NIfTI / PNG / JPEG) |
| **Output** | 4-class segmentation mask (BG, Whole Tumor, Core, Enhancing) |
| **Dice Score** | Whole Tumor: 91% · Tumor Core: 95% · Enhancing: 90% |
| **3D Visualization** | Plotly interactive tumor coordinate viewer |
| **PDF Reports** | Clinical-grade exportable reports (ReportLab) |
| **UI** | React + Canvas rendering · Blue header + Orange sidebar |

---

## Architecture (Wu et al. 2023)

```
Input MRI (256×256)
       │
   ┌── Stem Conv ──┐
   │               │
   ▼               ▼
Path 1 (3 conv)  Path 2 (5 conv)  Path 3 (7 conv)
Low-level        Mid-level        High-level
   │               │               │
   └───────────────┴───────────────┘
                    │
             Feature Fusion (concat + 1×1 conv)
                    │
          Global Attention Module
          ┌─────────────────┐
          │ Channel Attention│ (SE-style)
          │ Spatial Attention│ (CBAM)
          └─────────────────┘
                    │
           U-Net Decoder (skip connections)
                    │
         Output Mask (4 classes, 256×256)
```

**Classes:**
- `0` — Background
- `1` — Whole Tumor (green `#00C850`)
- `2` — Tumor Core (orange `#FF8C00`)
- `3` — Enhancing Tumor (yellow `#FFE600`)

---

## Quick Start

### Option A – Open Frontend Directly (No Server)
```bash
open brain-tumor-app/frontend/index.html
```
The dashboard loads with demo BraTS data instantly.

### Option B – Full Stack (Frontend + Backend)

```bash
cd brain-tumor-app
bash start.sh
```
Opens at: **http://localhost:8000**  
API Docs: **http://localhost:8000/docs**

### Option C – Install manually
```bash
cd brain-tumor-app/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Dashboard UI |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/demo` | Load BraTS demo analysis |
| `POST` | `/api/upload` | Upload MRI for segmentation |
| `POST` | `/api/report/pdf` | Generate PDF report |
| `POST` | `/api/report/json` | Export JSON report |
| `POST` | `/api/reanalyze` | Re-run analysis |

---

## Training

```bash
# Quick smoke-test with synthetic data
python backend/train.py --synthetic --epochs 5

# Full training (provide BraTS data directory)
python backend/train.py --data_dir /path/to/BraTS --epochs 50 --batch_size 4
```

---

## Project Structure

```
brain-tumor-app/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── model.py             # MultiPathFusionNet (PyTorch)
│   ├── inference.py         # Inference + preprocessing pipeline
│   ├── report_generator.py  # PDF report generation (ReportLab)
│   ├── train.py             # Training script
│   └── requirements.txt
├── frontend/
│   └── index.html           # React dashboard (CDN, no build step)
├── start.sh                 # One-command startup
└── README.md
```

---

## References

1. Wu, D., Qiu, S., Qin, J., & Zhao, P. (2023). Multi-Path Fusion Network Based Global Attention for Brain Tumor Segmentation. *Proceedings of ISAIMS 2023*.
2. Zhang, H. et al. (2024). Efficient Brain Tumor Segmentation with SSCN. *ACM TOMM*.
3. Yang, L. et al. (2019). Automatic Brain Tumor Segmentation Using Cascaded FCN. *IEEE ICCC*.
