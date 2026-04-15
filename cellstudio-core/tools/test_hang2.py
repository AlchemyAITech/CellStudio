from cellstudio.engine.config.config import Config
from cellstudio.datasets.registry import DATASET_REGISTRY
import cellstudio.datasets.segmentation
import cellstudio.pipeline.transforms.loading
import cellstudio.pipeline.transforms.visual_aug
import cellstudio.pipeline.transforms.formatting  # noqa: F401
from cellpose import dynamics
import time
import sys  # noqa: F401

cfg = Config.fromfile('configs/segmentation/cellpose_mido_seg.yaml')
ds = DATASET_REGISTRY.build(cfg.train_dataloader.dataset)

print(f"Loaded dataset with {len(ds)} images")

import cv2  # noqa: F401
import traceback

def test_hang():
    try:
        data = ds[0]
        mask = data['data_samples'].gt_instance_seg.numpy()
        
        print(f"Mask shape: {mask.shape}, max label: {mask.max()}")
        
        t0 = time.time()
        print("Running labels_to_flows...")
        out = dynamics.labels_to_flows([mask])[0]
        t1 = time.time()
        
        print(f"Successfully finished in {t1-t0:.4f}s")
        if len(out) != 4:
            print(f"WARNING: labels_to_flows output length is {len(out)}, expected 4")
            
    except Exception as e:
        print("ERROR:", str(e))
        traceback.print_exc()

test_hang()
