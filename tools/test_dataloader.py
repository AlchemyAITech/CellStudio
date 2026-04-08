import sys  # noqa: F401
import os  # noqa: F401

from cellstudio.engine.config.config import Config
from cellstudio.datasets.registry import DATASET_REGISTRY
from torch.utils.data import DataLoader
from functools import partial  # noqa: F401

# Necessary imports to register
import cellstudio.datasets.segmentation
import cellstudio.pipeline.transforms.loading
import cellstudio.pipeline.transforms.formatting
import cellstudio.pipeline.transforms.visual_aug  # noqa: F401

cfg = Config.fromfile('configs/segmentation/cellpose_mido_seg.yaml')
ds = DATASET_REGISTRY.build(cfg.train_dataloader.dataset)

def collate_fn(batch):
    from cellstudio.datasets.collate import default_collate
    return default_collate(batch)

dl = DataLoader(ds, batch_size=4, num_workers=0, collate_fn=collate_fn)

import time
print("Iterating DataLoader for first 5 batches...")
for i, data in enumerate(dl):
    t0 = time.time()
    print(f"Batch {i}: {data['imgs'].shape}")
    t1 = time.time()
    if i == 5: break
print("Done!")
