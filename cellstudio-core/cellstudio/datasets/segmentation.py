import os
import cv2
import torch
import numpy as np
from typing import Dict, Any, List

from .base import BaseDataset
from .registry import DATASET_REGISTRY
from .schema import CellDatasetConfig
from cellstudio.pipeline.compose import Compose

@DATASET_REGISTRY.register('CellposeSegmentationDataset')
class CellposeSegmentationDataset(BaseDataset):
    """
    Dataset loader for cellpose_standard.json formats.
    Outputs the image, binary semantic mask, and instances format for evaluation.
    """
    def __init__(self, 
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(data_root=data_root, ann_file=ann_file, **kwargs)
        self.data_root = data_root
        self.ann_file = os.path.join(data_root, ann_file) if not os.path.isabs(ann_file) else ann_file
        self.pipeline = Compose(pipeline) if pipeline else None
        
        self.data_list = self._load_data_list()
        
    def _load_data_list(self) -> List[Dict]:
        print(f"Loading annotations from {self.ann_file}...")
        dataset = CellDatasetConfig.load(self.ann_file)
        data_list = []
        for item in dataset.items:
            img_path = os.path.join(self.data_root, item.image_path)
            # Create a basic item representation
            data_info = {
                'img_path': img_path,
                'polygons': item.polygons,
                'img_id': os.path.basename(img_path)
            }
            if item.mask_path:
                data_info['mask_path'] = os.path.join(self.data_root, item.mask_path)
            data_list.append(data_info)
        return data_list

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data_info = self.data_list[idx].copy()
        img_path = data_info['img_path']
        
        # Load image via cv2.imdecode to avoid path issues
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
            
        h, w = img.shape[:2]
        data_info['img'] = img
        data_info['ori_shape'] = (h, w)
        
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        instance_mask = np.zeros((h, w), dtype=np.int32)
        
        bboxes = []
        labels = []
        
        for i, poly in enumerate(data_info['polygons']):
            pts = np.array(poly.points).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(semantic_mask, [pts], 1)
            cv2.fillPoly(instance_mask, [pts], i + 1)
            
            # Extract bounding boxes for anchor-based detectors like YOLO
            xmin, ymin = np.min(pts, axis=0)
            xmax, ymax = np.max(pts, axis=0)
            
            # Ensure valid boxes
            if xmax > xmin and ymax > ymin:
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(0) # MIDOG only has 1 class (mitotic figure) internally in semantic mask
            
        data_info['gt_semantic_seg'] = semantic_mask
        data_info['gt_instance_seg'] = instance_mask
        
        if bboxes:
            data_info['gt_bboxes'] = np.array(bboxes, dtype=np.float32)
            data_info['gt_labels'] = np.array(labels, dtype=np.int64)
            # Create a 3D tensor of instance masks [N, H, W] for Ultralytics
            n_inst = len(bboxes)
            masks_3d = np.zeros((n_inst, h, w), dtype=np.uint8)
            for i in range(n_inst):
                masks_3d[i] = (instance_mask == (i + 1))
            data_info['gt_masks'] = masks_3d
        else:
            data_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
            data_info['gt_labels'] = np.zeros((0,), dtype=np.int64)
            data_info['gt_masks'] = np.zeros((0, h, w), dtype=np.uint8)
        
        if self.pipeline:
            data_info = self.pipeline(data_info)
            
        return data_info
