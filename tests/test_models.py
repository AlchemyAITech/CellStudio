import os
import sys

# Ensure cellstudio is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellstudio.models.cellpose_plugin import CellposePlugin
from cellstudio.models.unet_plugin import UNetPlugin
import torch
import numpy as np

def test_cellpose():
    print("Testing CellposePlugin...")
    try:
        # Avoid downloading weights by just instantiating without heavy inference
        # or use cyto which is small
        plugin = CellposePlugin(model_type='cyto', device='cpu')
        
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        print(f"Running inference on dummy image of shape {dummy_image.shape}...")
        results = plugin.predict(dummy_image)
        print(f"Cellpose inference successful. Output mask shape: {results['masks'].shape}")
    except Exception as e:
        print(f"Cellpose tests failed: {e}")
        raise e

def test_unet():
    print("\nTesting UNetPlugin...")
    try:
        plugin = UNetPlugin(
            architecture='unet', 
            encoder_name='resnet18', # lightweight for testing
            encoder_weights=None,    # random weights for testing to avoid download 
            in_channels=3, 
            classes=2
        )
        dummy_tensor = torch.randn(1, 3, 256, 256)
        print(f"Running inference on dummy tensor of shape {dummy_tensor.shape}...")
        with torch.no_grad():
            output = plugin(dummy_tensor)
        print(f"UNet inference successful. Output mask shape: {output.shape}")
        
    except Exception as e:
        print(f"UNet tests failed: {e}")
        raise e

if __name__ == "__main__":
    print("========= Running Model Integration Tests =========")
    test_cellpose()
    test_unet()
    print("========= All Tests Passed =========")
