import sys  # noqa: F401
import os  # noqa: F401
import time
import numpy as np
from cellpose import dynamics
from scipy.ndimage import label
from cellstudio.engine.config.config import Config
from cellstudio.datasets.registry import DATASET_REGISTRY

# Necessary imports
import cellstudio.datasets.segmentation
import cellstudio.pipeline.transforms.loading
import cellstudio.pipeline.transforms.formatting
import cellstudio.pipeline.transforms.visual_aug  # noqa: F401

cfg = Config.fromfile('configs/segmentation/cellpose_mido_seg.yaml')
ds = DATASET_REGISTRY.build(cfg.train_dataloader.dataset)

print(f"Iterating all {len(ds)} dataset items to check for flow calculation hanging...")

for i in range(len(ds)):
    data = ds[i]
    mask_tensor = data['data_samples'].gt_instance_seg
    mask_np = np.ascontiguousarray(mask_tensor.numpy())
    
    if mask_np.max() == 0:
        continue
        
    clean_mask = np.zeros_like(mask_np)
    current_id = 1
    for inst_id in np.unique(mask_np):
        if inst_id == 0: continue
        inst_bin = (mask_np == inst_id)
        labeled_comps, num_comps = label(inst_bin)
        for comp_id in range(1, num_comps + 1):
            clean_mask[labeled_comps == comp_id] = current_id
            current_id += 1
            
    try:
        t0 = time.time()
        out = dynamics.labels_to_flows([clean_mask])[0]
        elapsed = time.time() - t0
        
        if elapsed > 2.0:
            print(f"WARN: Image {i} took {elapsed:.2f}s!")
            
    except Exception as e:
        print(f"CRASH on Image {i}: {e}")
        break
        
print("All instances completed without hanging!")
