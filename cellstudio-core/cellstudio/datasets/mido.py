import json
import os
from typing import List
import numpy as np
from .base import BaseDataset

class MIDODataset(BaseDataset):
    """
    Specialized Dataset for legacy MIDO-formatted Medical Images.
    Parses custom JSON structures with 'classes' and 'items'.
    """
    def _load_data_list(self) -> List[dict]:
        with open(self.ann_file, 'r', encoding='utf-8') as f:
            legacy_dict = json.load(f)
            
        categories = legacy_dict.get('classes', [])
        cat2idx = {cat: i for i, cat in enumerate(categories)}
        
        data_list = []
        for idx, item in enumerate(legacy_dict.get('items', [])):
            # Handle absolute paths from different old machines
            orig_path = item.get('image_path')
            basename = os.path.basename(orig_path)
            img_path = os.path.join(self.data_root, basename)
            
            # Skip entries where the image file doesn't exist on disk
            if not os.path.isfile(img_path):
                continue
            
            bboxes = []
            labels = []
            
            for box_item in item.get('bboxes', []):
                # The custom format uses xmin, ymin, xmax, ymax directly
                x1, y1, x2, y2 = box_item['xmin'], box_item['ymin'], box_item['xmax'], box_item['ymax']
                bboxes.append([x1, y1, x2, y2])
                
                cat_name = box_item['label']
                labels.append(cat2idx.get(cat_name, -1))
                
            data_info = {
                'img_path': img_path,
                'img_id': idx,
                'img_shape': (item.get('image_height', 0), item.get('image_width', 0))
            }
            
            if bboxes:
                data_info['gt_bboxes'] = np.array(bboxes, dtype=np.float32)
                data_info['gt_labels'] = np.array(labels, dtype=np.int64)
            else:
                data_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                data_info['gt_labels'] = np.zeros((0,), dtype=np.int64)
                
            data_list.append(data_info)
            
        return data_list
