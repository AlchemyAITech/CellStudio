"""
TileMIDODataset — Crops high-resolution pathology images into tiles with optional resize.

Each original image (e.g. 6577x4933) is split into tile_size tiles (e.g. 2048x2048),
then optionally resized to resize_to (e.g. 1024x1024) before entering the pipeline.
This preserves cell resolution: 50px cells at 2048->1024 become ~25px, still far
better than full-image resize which shrinks them to ~8px.
"""
import os
import json
import numpy as np
import cv2
import torch
from functools import lru_cache
from torch.utils.data import Dataset
from ..pipeline.compose import Compose
from .registry import DATASET_REGISTRY


@DATASET_REGISTRY.register('TileMIDODataset')
class TileMIDODataset(Dataset):
    """
    Tile-based MIDO dataset. Does NOT inherit from BaseDataset because it has
    a fundamentally different data loading pattern (tile grid instead of item list).
    """
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 tile_size: tuple = (2048, 2048),
                 resize_to: tuple = None,
                 stride: int = 2048,
                 min_bbox_area_ratio: float = 0.5,
                 min_boxes: int = 0,
                 pipeline: list = None,
                 **kwargs):
        self.data_root = data_root
        self.tile_size = tuple(tile_size) if not isinstance(tile_size, tuple) else tile_size
        self.resize_to = tuple(resize_to) if resize_to else None
        self.stride = stride
        self.min_bbox_area_ratio = min_bbox_area_ratio
        self.min_boxes = min_boxes
        self.pipeline = Compose(pipeline) if pipeline else None
        
        # Load annotations and build tile index
        ann_path = os.path.join(data_root, ann_file)
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.classes = data.get('classes', [])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Filter out items whose image files don't exist on disk
        all_items = data.get('items', [])
        self.raw_items = []
        skipped = 0
        for item in all_items:
            img_path = os.path.join(self.data_root, os.path.basename(item['image_path']))
            if os.path.exists(img_path):
                self.raw_items.append(item)
            else:
                skipped += 1
        if skipped > 0:
            print(f"TileMIDODataset: WARNING - skipped {skipped} items (image file not found)")
        
        self._build_tile_index()
    
    def _build_tile_index(self):
        """Pre-compute all tile positions and their GT boxes."""
        self.tiles = []
        tw, th = self.tile_size
        
        for item_idx, item in enumerate(self.raw_items):
            img_w = item.get('image_width', 0)
            img_h = item.get('image_height', 0)
            if img_w == 0 or img_h == 0:
                continue
            
            # Parse bboxes
            all_bboxes = []
            all_labels = []
            for b in item.get('bboxes', []):
                all_bboxes.append([b['xmin'], b['ymin'], b['xmax'], b['ymax']])
                all_labels.append(self.class_to_idx.get(b.get('label', ''), 0))
            
            all_bboxes = np.array(all_bboxes, dtype=np.float32).reshape(-1, 4)
            all_labels = np.array(all_labels, dtype=np.int64)
            
            # Generate tile grid
            for y0 in range(0, max(1, img_h - th + 1), self.stride):
                for x0 in range(0, max(1, img_w - tw + 1), self.stride):
                    y1 = min(y0 + th, img_h)
                    x1 = min(x0 + tw, img_w)
                    
                    if len(all_bboxes) > 0:
                        tile_bb, tile_lb = self._clip_boxes(all_bboxes, all_labels, x0, y0, x1, y1)
                    else:
                        tile_bb = np.zeros((0, 4), dtype=np.float32)
                        tile_lb = np.array([], dtype=np.int64)
                    
                    if len(tile_bb) < self.min_boxes:
                        continue
                    
                    self.tiles.append({
                        'item_idx': item_idx,
                        'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                        'gt_bboxes': tile_bb,
                        'gt_labels': tile_lb,
                    })
        
        resize_str = f" -> resize {self.resize_to}" if self.resize_to else ""
        print(f"TileMIDODataset: {len(self.raw_items)} images -> {len(self.tiles)} tiles "
              f"(tile={self.tile_size}{resize_str}, stride={self.stride}, min_boxes={self.min_boxes})")
    
    def _clip_boxes(self, bboxes, labels, x0, y0, x1, y1):
        """Clip bboxes to tile, keep those with enough overlap, translate to tile coords."""
        clipped = bboxes.copy()
        clipped[:, 0] = np.clip(bboxes[:, 0], x0, x1)
        clipped[:, 1] = np.clip(bboxes[:, 1], y0, y1)
        clipped[:, 2] = np.clip(bboxes[:, 2], x0, x1)
        clipped[:, 3] = np.clip(bboxes[:, 3], y0, y1)
        
        orig_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        clip_area = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
        
        valid = (clip_area > 0) & (orig_area > 0)
        ratio = np.where(orig_area > 0, clip_area / orig_area, 0)
        valid = valid & (ratio >= self.min_bbox_area_ratio)
        
        # Translate to tile-local coordinates
        result = clipped[valid]
        result[:, 0] -= x0; result[:, 1] -= y0
        result[:, 2] -= x0; result[:, 3] -= y0
        return result, labels[valid]
    
    def __len__(self):
        return len(self.tiles)
    
    def _load_image(self, item_idx):
        """Load and cache full-resolution image. Cache avoids re-reading
        the same ~30MB pathology image for each of its ~25 tiles."""
        if not hasattr(self, '_img_cache'):
            self._img_cache = {}
            self._cache_order = []
        
        if item_idx in self._img_cache:
            return self._img_cache[item_idx]
        
        # Evict oldest if cache is full (keep max 8 images in memory ~240MB)
        max_cache = 8
        if len(self._img_cache) >= max_cache:
            oldest = self._cache_order.pop(0)
            del self._img_cache[oldest]
        
        item = self.raw_items[item_idx]
        img_path = os.path.join(self.data_root, os.path.basename(item['image_path']))
        buf = np.fromfile(img_path, dtype=np.uint8)
        full_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if full_img is None:
            raise FileNotFoundError(f"Failed to load: {img_path}")
        full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
        
        self._img_cache[item_idx] = full_img
        self._cache_order.append(item_idx)
        return full_img
    
    def __getitem__(self, idx):
        tile = self.tiles[idx]
        item = self.raw_items[tile['item_idx']]
        tw, th = self.tile_size
        img_path = os.path.join(self.data_root, os.path.basename(item['image_path']))
        
        # Load from cache and crop
        full_img = self._load_image(tile['item_idx'])
        tile_img = full_img[tile['y0']:tile['y1'], tile['x0']:tile['x1']]
        
        # Pad edge tiles to full tile_size
        if tile_img.shape[0] < th or tile_img.shape[1] < tw:
            padded = np.zeros((th, tw, 3), dtype=tile_img.dtype)
            padded[:tile_img.shape[0], :tile_img.shape[1]] = tile_img
            tile_img = padded
        
        gt_bboxes = tile['gt_bboxes'].copy()
        
        # Resize tile and scale bboxes proportionally
        if self.resize_to is not None:
            rw, rh = self.resize_to
            sx = rw / tw
            sy = rh / th
            tile_img = cv2.resize(tile_img, (rw, rh), interpolation=cv2.INTER_LINEAR)
            if len(gt_bboxes) > 0:
                gt_bboxes[:, 0] *= sx  # xmin
                gt_bboxes[:, 1] *= sy  # ymin
                gt_bboxes[:, 2] *= sx  # xmax
                gt_bboxes[:, 3] *= sy  # ymax
            out_h, out_w = rh, rw
        else:
            out_h, out_w = th, tw
        
        results = {
            'img': tile_img,
            'img_path': img_path,
            'img_id': idx,
            'img_shape': (out_h, out_w),
            'ori_shape': (th, tw),
            'gt_bboxes': gt_bboxes,
            'gt_labels': tile['gt_labels'].copy(),
        }
        
        if self.pipeline is not None:
            results = self.pipeline(results)
        
        return results
