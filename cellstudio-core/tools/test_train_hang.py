import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
from scipy.ndimage import label
from tqdm import tqdm

from cellstudio.engine.config.config import Config
from cellstudio.tasks.registry import TASK_REGISTRY
import cellstudio.datasets.segmentation
import cellstudio.pipeline.transforms.loading
import cellstudio.pipeline.transforms.formatting
import cellstudio.models.adapters.cellpose_adapter  # noqa: F401

cfg = Config.fromfile('configs/segmentation/cellpose_mido_seg.yaml')
task = TASK_REGISTRY.build({'type': 'InstanceSegmentationTask', 'cfg': cfg})
task.build_datasets()

from cellpose import dynamics

dl = task.train_dataloader
print(f"Testing {len(dl)} batches in training set for deadlocks...")

for i, batch in enumerate(tqdm(dl)):
    mask_list = []
    for ds in batch['data_samples']:
        if hasattr(ds, 'gt_instance_seg'):
            mask_list.append(ds.gt_instance_seg)
            
    for mask in mask_list:
        mask_np = np.ascontiguousarray(mask.cpu().numpy())
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
                
        # This is where it potentially hangs
        out = dynamics.labels_to_flows([clean_mask], device=torch.device('cpu'))[0]

print("Successfully verified all masks without deadlock!")
