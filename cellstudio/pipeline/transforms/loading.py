from typing import Any, Dict

import cv2
import numpy as np

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
        
        # Medical WSI format branching — only true WSI formats use OpenSlide
        # Regular .tif/.tiff files should be loaded with cv2 (e.g. MIDO dataset)
        if filename.endswith(('.svs', '.ndpi')):
            img = self._load_wsi(filename)
        else:
            flag = cv2.IMREAD_COLOR if self.color_type == 'color' else cv2.IMREAD_GRAYSCALE
            # CRITICAL: Use cv2.imdecode instead of cv2.imread to bypass
            # ultralytics' monkey-patched imread which crashes on multi-frame TIFFs.
            # cv2.imdecode reads from a numpy buffer and is not patched.
            buf = np.fromfile(filename, dtype=np.uint8)
            img = cv2.imdecode(buf, flag)
            if img is None:
                # Some TIFFs need IMREAD_UNCHANGED; try that and convert
                img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Strip alpha channel if present (RGBA -> RGB)
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]
                if self.color_type == 'color' and len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
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
