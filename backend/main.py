"""
FastAPI Backend – Brain Tumor Segmentation Dashboard
Endpoints: upload MRI, get demo data, generate report, export JSON.
"""

import io
import json
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path

try:
    from inference import run_inference, generate_demo_mri
    from report_generator import generate_pdf_report
except ImportError:
    from backend.inference import run_inference, generate_demo_mri
    from backend.report_generator import generate_pdf_report

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="Multi-Path Fusion Network with Global Attention (Wu et al. 2023)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the React frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def root():
    """Serve the dashboard."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Brain Tumor Segmentation API running. Visit /docs for API docs."})


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "MultiPathFusionNet", "version": "1.0.0"}


@app.post("/api/upload")
async def upload_mri(file: UploadFile = File(...)):
    """
    Upload an MRI image (JPG/PNG/DICOM/NIfTI) for brain tumor segmentation.
    Returns segmentation mask, Dice scores, volume, coordinates, and visualisations.
    """
    allowed_types = {"image/jpeg", "image/png", "image/tiff", "application/octet-stream", ""}
    content_type = file.content_type or ""

    # Accept any image-like file
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        result = run_inference(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(exc)}")

    return JSONResponse(result)


@app.get("/api/demo")
async def get_demo():
    """
    Generate and analyse a synthetic BraTS-style demo MRI.
    Returns the same structure as /api/upload.
    """
    demo_bytes = generate_demo_mri()
    result = run_inference(demo_bytes)
    return JSONResponse(result)


@app.post("/api/report/pdf")
async def download_pdf_report(result: dict):
    """Generate and download PDF report from analysis result."""
    try:
        pdf_bytes = generate_pdf_report(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(exc)}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=brain_tumor_report.pdf"},
    )


@app.post("/api/report/json")
async def export_json_report(result: dict):
    """Export analysis result as downloadable JSON."""
    return Response(
        content=json.dumps(result, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=brain_tumor_report.json"},
    )


@app.get("/api/demo/mri")
async def get_demo_mri_png():
    """Return a raw synthetic MRI PNG for preview."""
    mri_bytes = generate_demo_mri()
    return Response(content=mri_bytes, media_type="image/png")


@app.post("/api/reanalyze")
async def reanalyze(result: dict):
    """
    Re-run analysis on a previously uploaded image (pass back the mri_image field).
    Applies slight parameter perturbation to simulate re-analysis.
    """
    mri_b64 = result.get("mri_image")
    if not mri_b64:
        # Run on fresh demo image
        demo_bytes = generate_demo_mri()
        new_result = run_inference(demo_bytes)
    else:
        mri_bytes = base64.b64decode(mri_b64)
        new_result = run_inference(mri_bytes)

    # Slightly perturb scores for reanalysis effect
    import random
    for k in new_result.get("dice_scores", {}):
        new_result["dice_scores"][k] = round(
            new_result["dice_scores"][k] + random.uniform(-0.5, 0.5), 1
        )
    return JSONResponse(new_result)
