import torch
import sys  # noqa: F401
import logging
from cellstudio.models.adapters.cellpose_adapter import CellposeSegAdapter

logging.basicConfig(level=logging.INFO)

print("Init model...")
model = CellposeSegAdapter(model_type='cyto3', diam_mean=30)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

imgs = torch.randn(4, 3, 256, 256).to(device)

class MockSample:
    def __init__(self):
        self.gt_instance_seg = torch.zeros(256, 256, dtype=torch.int32)

data_samples = [MockSample() for _ in range(4)]

opt = torch.optim.Adam(model.parameters(), lr=0.001)

print("Forward training...")
import time
t0 = time.time()
res = model.forward_train(imgs, data_samples)
print(f"Forward pass took: {time.time()-t0:.4f}s")

t0 = time.time()
print("Backward training...")
loss = res['loss']
opt.zero_grad()
loss.backward()
opt.step()
t1 = time.time()
print(f"Backward pass took: {t1-t0:.4f}s")
