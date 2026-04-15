import os
import json
from collections import defaultdict
from pathstudio.datasets.converters.base import BaseDataConverter
from pathstudio.datasets.schema import CellDataItem, BBox, CellDatasetConfig

class MidoConverter(BaseDataConverter):
    """
    Converter for the MIDO dataset (Mitotic Figures).
    Assumes standard COCO-like MIDOGpp.json format where:
    images: [{id, file_name, width, height}]
    annotations: [{image_id, bbox: [x, y, w, h] or [xmin, ymin, xmax, ymax], category_id}]
    categories: [{id, name}]
    """
    
    def convert(self) -> str:
        json_path = os.path.join(self.raw_data_dir, "MIDOGpp.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotations file not found at {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
            
        # Map category id to name
        cat_map = {cat['id']: cat.get('name', str(cat['id'])) 
                   for cat in coco_data.get('categories', [])}
        
        # If no explicit categories, fallback to generic mappings
        if not cat_map:
            cat_map = {1: "mitotic_figure", 2: "hard_negative"}

        # Map image id to filepath
        img_dict = {}
        for img in coco_data.get('images', []):
            abs_path = os.path.abspath(os.path.join(self.raw_data_dir, img['file_name']))
            img_dict[img['id']] = {
                "path": abs_path,
                "w": img.get('width'),
                "h": img.get('height')
            }
            
        # Group annotations by image_id
        annos_by_img = defaultdict(list)
        for ann in coco_data.get('annotations', []):
            annos_by_img[ann['image_id']].append(ann)
            
        items = []
        for img_id, img_info in img_dict.items():
            bboxes = []
            for ann in annos_by_img.get(img_id, []):
                bbox = ann['bbox']
                # MIDOG bbox format: [xmin, ymin, xmax, ymax]
                # If COCO standard it is [x, y, w, h], we treat based on common MIDOG formats
                # The bbox arrays show xmin, ymin, xmax, ymax based on observation
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                
                label_name = cat_map.get(ann.get('category_id'), "unknown")
                
                bboxes.append(BBox(
                    xmin=float(xmin),
                    ymin=float(ymin),
                    xmax=float(xmax),
                    ymax=float(ymax),
                    label=label_name
                ))
                
            items.append(CellDataItem(
                image_path=img_info["path"],
                image_width=img_info["w"],
                image_height=img_info["h"],
                bboxes=bboxes,
                metadata={"source": "MIDOGpp"}
            ))
            
        config = CellDatasetConfig(items=items)
        return self.save_standardized_json(config, "mido_standard.json")
