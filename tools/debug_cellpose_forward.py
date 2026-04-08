import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys  # noqa: F401
import torch  # noqa: F401
import numpy as np  # noqa: F401

# Mocking
from cellstudio.engine.config.config import Config
from cellstudio.tasks.registry import TASK_REGISTRY

# Imports
import cellstudio.datasets.segmentation
import cellstudio.pipeline.transforms.loading
import cellstudio.pipeline.transforms.formatting
import cellstudio.models.adapters.cellpose_adapter  # noqa: F401

cfg = Config.fromfile('configs/segmentation/cellpose_mido_seg.yaml')
task = TASK_REGISTRY.build({'type': 'InstanceSegmentationTask', 'cfg': cfg})

print("Building Datasets...")
task.build_datasets()
print("Building Model...")
task.build_model()

print("Getting a batch...")
batch = next(iter(task.train_dataloader))
print(f"Batch received! Images shape: {batch['imgs'].shape}")

print("Running Forward Pass...")
task.model.to('cuda')
batch['imgs'] = batch['imgs'].to('cuda')
res = task.model.forward_train(batch['imgs'], batch['data_samples'])
print(f"Finished Forward Pass! Loss: {res['loss'].item()}")

print("Testing backward pass...")
res['loss'].backward()
print("SUCCESSFULLY PASSED BACKWARD()")

