import torch
from torch.utils.data.dataloader import default_collate
from typing import Sequence
import collections.abc

def pseudo_collate(batch: Sequence[dict]) -> dict:
    """
    Zenith Collation Engine.
    Crucial fix for medical batching: unlike default PyTorch collate which attempts to stack all Tensors 
    (and crashes when instances like bounding boxes/masks vary in length between images), 
    this safely stacks the guaranteed fixed-dimension properties (like exactly augmented images), 
    while preserving the variable-length structured properties (like DataSamples) as accessible lists.
    """
    if not isinstance(batch[0], collections.abc.Mapping):
        return default_collate(batch)
        
    collated = {}
    for key in batch[0]:
        if key == 'imgs':
            # Images should all be exactly the same (N, C, H, W) shape post-augmentation DAG
            collated[key] = torch.stack([d[key] for d in batch])
        elif key == 'data_samples':
            # Bypass stack: DataSamples house variable number bounding boxes and non-tensor instances
            collated[key] = [d[key] for d in batch]
        else:
            try:
                collated[key] = default_collate([d[key] for d in batch])
            except TypeError:
                collated[key] = [d[key] for d in batch]
                
    return collated
