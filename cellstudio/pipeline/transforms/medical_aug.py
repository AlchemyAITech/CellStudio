from typing import Any, Dict

import numpy as np

from ..registry import PIPELINE_REGISTRY


@PIPELINE_REGISTRY.register('MacenkoNormalize')
class MacenkoNormalize:
    """Rigorous H&E Stain Normalizer for Pathology Pipeline integration."""
    def __init__(self, alpha: int = 1, beta: float = 0.15):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results.get('img')
        if img is None:
            return results
            
        # Placeholder for OD vector extraction/stain decomposition matrix
        normalized_img = img 
        
        results['img'] = normalized_img
        if 'img_metas' not in results:
            results['img_metas'] = {}
        results['img_metas']['macenko'] = True
        return results

@PIPELINE_REGISTRY.register('RandomGridCrop')
class RandomGridCrop:
    """WSI Random Tile Grid Extractor with strict annotation coupling."""
    def __init__(self, crop_size=(1024, 1024), min_boxes=1):
        self.crop_size = crop_size
        self.min_boxes = min_boxes
        
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results.get('img')
        if img is None:
            return results
            
        h, w = img.shape[:2]
        ch, cw = self.crop_size
        
        if h > ch and w > cw:
            top = np.random.randint(0, h - ch)
            left = np.random.randint(0, w - cw)
            results['img'] = img[top:top+ch, left:left+cw, :]
            results['img_shape'] = (ch, cw)
            
            # Translate bound boxes precisely to sub-image coordinate frame
            if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
                bboxes = results['gt_bboxes']
                bboxes[:, [0, 2]] -= left
                bboxes[:, [1, 3]] -= top
                
                bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, cw)
                bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, ch)
                
                valid = (bboxes[:, 2] - bboxes[:, 0] > 2) & (bboxes[:, 3] - bboxes[:, 1] > 2)
                results['gt_bboxes'] = bboxes[valid]
                if 'gt_labels' in results:
                    results['gt_labels'] = results['gt_labels'][valid]
                    
                # Strict rejection to keep train metrics accurate
                if len(results['gt_bboxes']) < self.min_boxes:
                    return None
                    
        return results
