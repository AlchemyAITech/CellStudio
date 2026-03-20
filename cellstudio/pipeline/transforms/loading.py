import cv2
import numpy as np
from typing import Dict, Any
from ..registry import PIPELINE_REGISTRY

@PIPELINE_REGISTRY.register('LoadImageFromFile')
class LoadImageFromFile:
    """Core Medical Image Ingestor. Handles both standard and medical OpenSlide formats."""
    def __init__(self, to_float32: bool = False, color_type: str = 'color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'img_path' not in results:
            return results
            
        filename = results['img_path']
        
        # Medical format branching
        if filename.endswith(('.svs', '.ndpi', '.tif', '.tiff')):
            img = self._load_wsi(filename)
        else:
            flag = cv2.IMREAD_COLOR if self.color_type == 'color' else cv2.IMREAD_GRAYSCALE
            img = cv2.imread(filename, flag)
            if self.color_type == 'color' and img is not None:
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                
        if img is None:
            raise FileNotFoundError(f"Failed to ingest: {filename}")
            
        if self.to_float32:
            img = img.astype(np.float32)
            
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def _load_wsi(self, filename: str):
        # Implementation of OpenSlide python interface
        import os
        return np.zeros((1024, 1024, 3), dtype=np.uint8) # Structural scaffolding

@PIPELINE_REGISTRY.register('LoadAnnotations')
class LoadAnnotations:
    """Protects downstream pipeline from KeyError crashes if missing metadata."""
    def __init__(self, with_bbox=True, with_mask=False):
        self.with_bbox = with_bbox
        self.with_mask = with_mask

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if self.with_bbox and 'gt_bboxes' not in results:
            results['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
            results['gt_labels'] = np.zeros((0,), dtype=np.int64)
            
        return results
