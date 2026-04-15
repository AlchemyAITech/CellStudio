import json
import os
from typing import Dict, Any
from .base import BaseConverter
from ...structures.csuos import (
    CSUFeatureCollection, CSUFeature, CSUProperties, CSUGeometry, 
    CSUClassProperties, CSUImageContext, CSUMetadata
)
from PIL import Image

class LegacyConverter(BaseConverter):
    """Converts the early custom CellStudio dict {"classes": [], "items": [{"bboxes", "polygons"}] } into UDF."""
    def parse_to_udf(self, source_path: str, data_root: str = "", **kwargs) -> CSUFeatureCollection:
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        class_names = data.get('classes', [])
        items = data.get('items', [])
        
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        
        features = []
        image_contexts = []
        
        for idx, item in enumerate(items):
            img_path = item.get('image_path', '')
            img_name = os.path.basename(img_path)
            img_id = img_name.split('.')[0]
            
            w = item.get('image_width', 0)
            h = item.get('image_height', 0)
            if (w == 0 or h == 0) and data_root:
                full_p = os.path.join(data_root, img_path)
                if os.path.exists(full_p):
                    try:
                        with Image.open(full_p) as im:
                            w, h = im.size
                    except: pass
            
            task_t = 'classification'
            if item.get('bboxes'): task_t = 'detection'
            if item.get('polygons'): task_t = 'segmentation'
            
            ctx = CSUImageContext(image_id=img_id, file_path=img_path, width=w, height=h, task_type=task_t)
            image_contexts.append(ctx)
            
            # --- Parsing Classification Labels ---
            if task_t == 'classification':
                for label in item.get('cls_labels', []):
                    cat_id = class_to_id.get(label, 0)
                    geom = CSUGeometry(type="BBox", coordinates=[0, 0, w, h])
                    props = CSUProperties(
                        object_type="image",
                        classification=CSUClassProperties(name=label, id=cat_id)
                    )
                    feat = CSUFeature(geometry=geom, properties=props, image_id=img_id, id=f"cls_{img_id}_{cat_id}")
                    features.append(feat)
                    
            # --- Parsing Bounding Boxes ---
            for b_idx, bbox in enumerate(item.get('bboxes', [])):
                lbl = bbox.get('label', 'unknown')
                coords = bbox.get('bbox', [0,0,0,0]) # [x1, y1, x2, y2]
                geom = CSUGeometry(type="BBox", coordinates=coords)
                props = CSUProperties(
                    object_type="detection",
                    classification=CSUClassProperties(name=lbl, id=class_to_id.get(lbl, 0))
                )
                feat = CSUFeature(geometry=geom, properties=props, image_id=img_id, id=f"det_{img_id}_{b_idx}")
                features.append(feat)
                
            # --- Parsing Polygons ---
            for p_idx, poly in enumerate(item.get('polygons', [])):
                lbl = poly.get('label', 'unknown')
                flat_pts = poly.get('points', [])
                
                # Convert [x1,y1,  x2,y2, ...] into [[x1,y1], [x2,y2]]
                pts = []
                for i in range(0, len(flat_pts), 2):
                    if i+1 < len(flat_pts):
                        pts.append([flat_pts[i], flat_pts[i+1]])
                if pts and pts[0] != pts[-1]: pts.append(pts[0])
                
                geom = CSUGeometry(type="Polygon", coordinates=[pts])
                props = CSUProperties(
                    object_type="cell",  # Assign custom segmentations to "cell"
                    classification=CSUClassProperties(name=lbl, id=class_to_id.get(lbl, 0))
                )
                feat = CSUFeature(geometry=geom, properties=props, image_id=img_id, id=f"seg_{img_id}_{p_idx}")
                features.append(feat)
                
        meta = CSUMetadata(generator="LegacyFormat_To_UDF")
        return CSUFeatureCollection(features=features, metadata=meta, images=image_contexts)
        
    def export_from_udf(self, udf_collection: CSUFeatureCollection, output_path: str, **kwargs) -> Any:
        pass
