import torch
from backend.model import MultiPathFusionNet

ckpt = torch.load('/Users/srivardhan/Downloads/best_model.pth', map_location='cpu', weights_only=False)
sd = ckpt['model_state']
print('Checkpoint info:')
print('  Epoch:', ckpt['epoch'])
print('  Dice:', ckpt.get('tumor_dice', 'N/A'))
print('  head.weight shape:', sd['head.weight'].shape)
print('  head.bias shape:', sd['head.bias'].shape)
print('  stem input channels:', sd['stem.block.0.weight'].shape)

# Try loading into 4-class model
model4 = MultiPathFusionNet(in_channels=1, num_classes=4)
try:
    model4.load_state_dict(sd, strict=True)
    print('\nStrict load into 4-class model: SUCCESS')
except Exception as e:
    print(f'\nStrict load into 4-class model: FAILED - {e}')

# Try loading into 2-class model
model2 = MultiPathFusionNet(in_channels=1, num_classes=2)
try:
    model2.load_state_dict(sd, strict=True)
    print('Strict load into 2-class model: SUCCESS')
except Exception as e:
    print(f'Strict load into 2-class model: FAILED - {e}')

# Try partial load (everything except head)
model4b = MultiPathFusionNet(in_channels=1, num_classes=4)
missing, unexpected = model4b.load_state_dict(sd, strict=False)
print(f'\nPartial load into 4-class: missing={missing}, unexpected={unexpected}')
