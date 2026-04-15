import os
from typing import List
from .base import BaseDataset
from .schema import CellDatasetConfig
from .registry import DatasetRegistry

@DatasetRegistry.register('StandardClassificationDataset')
class StandardClassificationDataset(BaseDataset):
    """
    Consumes the standard CellDatasetConfig JSON explicitly mapped cls_labels to integer labels.
    """
    def __init__(self, class_map: dict = None, **kwargs):
        self.class_map = class_map or {}
        # Ensure data_prefix is passed if omitted
        if 'data_prefix' not in kwargs:
            kwargs['data_prefix'] = dict(img_path='')
        super().__init__(**kwargs)
        
    def _load_data_list(self) -> List[dict]:
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")
            
        dataset_config = CellDatasetConfig.load(self.ann_file)
        
        # If class_map wasn't provided, use the schema auto-extracted properties
        if not self.class_map:
            self.class_map = {cls_name: i for i, cls_name in enumerate(dataset_config.classes)}
            
        self.classes = list(self.class_map.keys())
        
        data_list = []
        for item in dataset_config.items:
            # We strictly assume single-label classification here
            if not item.cls_labels:
                continue
                
            label_name = item.cls_labels[0]
            if label_name not in self.class_map:
                continue
                
            img_path = item.image_path
            if not os.path.isabs(img_path):
                img_path = os.path.join(self.data_root, img_path)
            
            data_info = {
                'img_path': img_path,
                'gt_label': self.class_map[label_name]
            }
            data_list.append(data_info)
            
        print(f"[{self.__class__.__name__}] Loaded {len(data_list)} items from {self.ann_file}. Classes: {self.class_map}")
        return data_list
