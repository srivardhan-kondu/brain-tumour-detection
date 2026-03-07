import sys
sys.path.insert(0, '.')

from model import MultiPathFusionNet
import torch

model = MultiPathFusionNet(in_channels=1, num_classes=4)
model.eval()
x = torch.randn(1, 1, 256, 256)
with torch.no_grad():
    y = model(x)
params = sum(p.numel() for p in model.parameters())
print('Model OK')
print('  Input  shape:', tuple(x.shape))
print('  Output shape:', tuple(y.shape))
print('  Parameters  :', f'{params:,}')

from inference import run_inference, generate_demo_mri
demo_bytes = generate_demo_mri()
print('Demo MRI generated:', len(demo_bytes), 'bytes')

result = run_inference(demo_bytes)
print('Inference complete')
dice = result['dice_scores']
print('  Dice WT={}% TC={}% ET={}'.format(
    dice['whole_tumor'], dice['tumor_core'], dice['enhancing_tumor']))
print('  Volume    :', result['tumor_volume_mm3'], 'mm3')
print('  Confidence:', result['confidence_label'], result['confidence'])
print('  Overlay len:', len(result.get('overlay_image', '')))
print('ALL TESTS PASSED')
