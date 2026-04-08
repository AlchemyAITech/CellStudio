import numpy as np
import cv2
import torch
from ..registry import PIPELINE_REGISTRY

@PIPELINE_REGISTRY.register('RandomFlip')
class RandomFlip:
    def __init__(self, prob=0.5, direction='horizontal'):
        self.prob = prob
        self.direction = direction
    def __call__(self, results):
        if 'img' in results and np.random.rand() < self.prob:
            img = results['img']
            h, w = img.shape[:2]
            flip_code = 1 if self.direction == 'horizontal' else 0
            results['img'] = cv2.flip(img, flip_code)
            
            if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
                bboxes = results['gt_bboxes']
                if self.direction == 'horizontal':
                    # [x1, y1, x2, y2] -> [w-x2, y1, w-x1, y2]
                    bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
                else: # vertical
                    # [x1, y1, x2, y2] -> [x1, h-y2, x2, h-y1]
                    bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]
                
                # Filter invalid boxes just in case
                valid_mask = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
                results['gt_bboxes'] = bboxes[valid_mask]
                if 'gt_labels' in results:
                    results['gt_labels'] = results['gt_labels'][valid_mask]
                if 'gt_masks' in results and len(results['gt_masks']) > 0:
                    masks = results['gt_masks']
                    flipped_masks = np.zeros_like(masks)
                    for i in range(masks.shape[0]):
                        flipped_masks[i] = cv2.flip(masks[i], flip_code)
                    results['gt_masks'] = flipped_masks[valid_mask]
                    
            if 'gt_semantic_seg' in results:
                results['gt_semantic_seg'] = cv2.flip(results['gt_semantic_seg'], flip_code)
            if 'gt_instance_seg' in results:
                results['gt_instance_seg'] = cv2.flip(results['gt_instance_seg'], flip_code)
                
        return results

@PIPELINE_REGISTRY.register('ColorJitter')
class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast
    def __call__(self, results):
        if 'img' in results:
            img = results['img'].astype(np.float32)
            alpha = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            beta = 255.0 * np.random.uniform(-self.brightness, self.brightness)
            img = np.clip(img * alpha + beta, 0, 255)
            results['img'] = img
        return results

@PIPELINE_REGISTRY.register('Resize')
class Resize:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, results):
        if 'img' in results:
            img = results['img']
            h_old, w_old = img.shape[:2]
            w_new, h_new = self.size
            results['img'] = cv2.resize(img, (w_new, h_new))
            results['img_shape'] = results['img'].shape
            
            w_scale = w_new / w_old
            h_scale = h_new / h_old
            
            if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
                bboxes = results['gt_bboxes']
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w_scale
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h_scale
                # Clip to new edges
                bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, w_new)
                bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, h_new)
                
                # Eradicate collapsed zero-width or zero-height bounding boxes
                valid_mask = (bboxes[:, 2] > bboxes[:, 0] + 1.0) & (bboxes[:, 3] > bboxes[:, 1] + 1.0)
                results['gt_bboxes'] = bboxes[valid_mask]
                if 'gt_labels' in results:
                    results['gt_labels'] = results['gt_labels'][valid_mask]
                if 'gt_masks' in results and len(results['gt_masks']) > 0:
                    masks = results['gt_masks']
                    resized_masks = np.zeros((masks.shape[0], h_new, w_new), dtype=masks.dtype)
                    for i in range(masks.shape[0]):
                        resized_masks[i] = cv2.resize(masks[i], (w_new, h_new), interpolation=cv2.INTER_NEAREST)
                    results['gt_masks'] = resized_masks[valid_mask]
                    
            if 'gt_semantic_seg' in results:
                results['gt_semantic_seg'] = cv2.resize(results['gt_semantic_seg'], (w_new, h_new), interpolation=cv2.INTER_NEAREST)
            if 'gt_instance_seg' in results:
                results['gt_instance_seg'] = cv2.resize(results['gt_instance_seg'], (w_new, h_new), interpolation=cv2.INTER_NEAREST)
                
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
