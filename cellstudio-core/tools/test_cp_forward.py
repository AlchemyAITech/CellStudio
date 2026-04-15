import torch
import sys  # noqa: F401
import logging
from cellstudio.models.adapters.cellpose_adapter import CellposeSegAdapter

logging.basicConfig(level=logging.INFO)

print("Init model...")
model = CellposeSegAdapter(model_type='cyto3', diam_mean=30)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create mock data
imgs = torch.randn(4, 3, 256, 256).to(device)

class MockSample:
    def __init__(self):
        self.gt_instance_seg = torch.zeros(256, 256, dtype=torch.int32)

data_samples = [MockSample() for _ in range(4)]

print("Forward training...")
import time
t0 = time.time()
res = model.forward_train(imgs, data_samples)
t1 = time.time()
print(f"Loss dict: {res}")
print(f"Time taken: {t1-t0:.4f}s")
