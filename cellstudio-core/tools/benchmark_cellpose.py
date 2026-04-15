import numpy as np
import torch
import time
from cellpose import dynamics

# Simulate a batch of 4 masks, 512x512
masks = []
for _ in range(4):
    mask = np.zeros((256, 256), dtype=np.int32)
    # create some random cells
    for i in range(1, 101): # 100 cells
        x, y = np.random.randint(10, 240, 2)
        r = np.random.randint(5, 15)
        Y, X = np.ogrid[:256, :256]
        dist = (X - x)**2 + (Y - y)**2
        mask[dist <= r**2] = i
    masks.append(torch.from_numpy(mask))

t0 = time.time()
print("Starting labels_to_flows...")
out_flows = []
for m in masks:
    m_np = m.cpu().numpy()
    out = dynamics.labels_to_flows([m_np])[0]
    out_flows.append(out)
t1 = time.time()
print(f"Time for 4 masks: {t1-t0:.4f} seconds")
