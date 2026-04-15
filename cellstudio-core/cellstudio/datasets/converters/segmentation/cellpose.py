import os
import glob
from pathstudio.datasets.converters.base import BaseDataConverter
from pathstudio.datasets.schema import CellDataItem, CellDatasetConfig

class CellposeConverter(BaseDataConverter):
    """
    Converter for the Cellpose segmentation dataset format.
    Assumes a flat directory structure where images end with '_img.png' 
    and their corresponding masks end with '_masks.png' or similar suffixes.
    """
    
    def convert(self) -> str:
        # We will scan the 'train' and 'test' folders if they exist
        img_suffix = "_img.png"
        mask_suffix = "_masks.png"
        
        search_dirs = [
            os.path.join(self.raw_data_dir, "train"),
            os.path.join(self.raw_data_dir, "test"),
            self.raw_data_dir  # Fallback to root if unstructured
        ]
        
        items = []
        for target_dir in search_dirs:
            if not os.path.exists(target_dir):
                continue
                
            img_paths = glob.glob(os.path.join(target_dir, f"*{img_suffix}"))
            
            for img_path in img_paths:
                base_prefix = img_path[:-len(img_suffix)]
                mask_path = f"{base_prefix}{mask_suffix}"
                
                # Check if corresponding mask exists
                if not os.path.exists(mask_path):
                    continue
                    
                import cv2
                import numpy as np
                from pathstudio.datasets.schema import Polygon
                
                polygons = []
                image_width, image_height = None, None
                
                mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask_img is not None:
                    image_height, image_width = mask_img.shape[:2]
                    instances = np.unique(mask_img)
                    for inst_id in instances:
                        if inst_id == 0: continue
                        inst_mask = (mask_img == inst_id).astype(np.uint8)
                        contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if len(contour) >= 3:
                                pts = contour.flatten().astype(float).tolist()
                                polygons.append(Polygon(label="cell", points=pts))
                                
                items.append(CellDataItem(
                    image_path=os.path.abspath(img_path),
                    image_width=image_width,
                    image_height=image_height,
                    mask_path=os.path.abspath(mask_path),
                    cls_labels=["cell"],
                    polygons=polygons,
                    metadata={"source": "cellpose"}
                ))
                
        if not items:
            raise ValueError(f"No valid image/mask pairs found with suffixes {img_suffix}/{mask_suffix} in {self.raw_data_dir}")
            
        dataset_config = CellDatasetConfig(items=items)
        return self.save_standardized_json(dataset_config, "cellpose_standard.json")
