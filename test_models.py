"""Test script to verify model loading"""
import sys
import torch
import yaml

# Test AASIST model
print("=" * 50)
print("Testing AASIST-L Model")
print("=" * 50)

try:
    from models.aasist_model import Model as AASIST
    
    # Load config
    with open('models/model_config_AASIST.yaml') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = AASIST(config['model'])
    print("✓ AASIST model initialized")
    
    # Load checkpoint
    checkpoint = torch.load('weights/AASIST-L.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    print("✓ AASIST-L checkpoint loaded successfully")
    print(f"  Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test RawNet model
print("\n" + "=" * 50)
print("Testing RawNet (Vocoder) Model")
print("=" * 50)

try:
    from models.vocoder_model import RawNet
    
    # Load config
    with open('models/model_config_RawNet.yaml') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = RawNet(config['model'], 'cpu')
    print("✓ RawNet model initialized")
    
    # Load checkpoint
    checkpoint = torch.load('weights/librifake_pretrained_lambda0.5_epoch_25.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    print("✓ RawNet checkpoint loaded successfully")
    print(f"  Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("All tests completed!")
print("=" * 50)
