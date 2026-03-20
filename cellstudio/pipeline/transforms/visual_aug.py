import numpy as np
import cv2
import torch
from ..registry import PIPELINE_REGISTRY

@PIPELINE_REGISTRY.register('Resize')
class Resize:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, results):
        if 'img' in results:
            img = results['img']
            # size is [W, H] for cv2
            results['img'] = cv2.resize(img, tuple(self.size))
            results['img_shape'] = results['img'].shape
        return results

@PIPELINE_REGISTRY.register('Normalize')
class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        
    def __call__(self, results):
        if 'img' in results:
            img = results['img'].astype(np.float32)
            img = (img - self.mean) / self.std
            results['img'] = img
        return results

@PIPELINE_REGISTRY.register('PackInputs')
class PackInputs:
    def __init__(self, keys=['img']):
        self.keys = keys
        
    def __call__(self, results):
        packed = {'data_samples': {k: v for k, v in results.items() if k not in self.keys}}
        if 'img' in results:
            img = results['img']
            if len(img.shape) == 3:
                img = img.transpose(2, 0, 1) # HWC to CHW
            packed['imgs'] = torch.from_numpy(img).contiguous()
        return packed
