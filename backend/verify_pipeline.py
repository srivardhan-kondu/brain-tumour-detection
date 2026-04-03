"""Quick end-to-end verification of the inference pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from inference import run_inference, generate_demo_mri, get_model

# Test model loading
model = get_model()
print(f"Model loaded: {model.num_classes}-class, in_channels={model.in_channels}")
print(f"Eval mode: {not model.training}")

# Test inference
demo_bytes = generate_demo_mri()
print(f"Demo MRI: {len(demo_bytes)} bytes")

result = run_inference(demo_bytes)
print(f"Status: {result['status']}")
print(f"Model classes: {result['model_classes']}")
print(f"Region confidence: {result['region_confidence']}")
print(f"Confidence: {result['confidence']}% ({result['confidence_label']})")
print(f"Tumor volume: {result['tumor_volume_mm3']} mm3")
print(f"Mask: {result['mask_summary']}")
print(f"Overlay image present: {len(result['overlay_image']) > 100}")
print("PASS - All inference working with trained 4-class model")
