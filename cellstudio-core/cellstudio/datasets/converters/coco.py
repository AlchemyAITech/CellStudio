import json
import os
from typing import Dict, Any

from .base import BaseConverter
from ...structures.csuos import (
    CSUFeatureCollection, CSUFeature, CSUProperties, CSUGeometry, 
    CSUClassProperties, CSUImageContext, CSUMetadata
)

class COCOConverter(BaseConverter):
    """Parses standard MS COCO JSON annotations into CellStudio Universal Data Foundation format."""
    
    def parse_to_udf(self, source_path: str, **kwargs) -> CSUFeatureCollection:
        with open(source_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
            
        # 1. Map Categories
        categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
        
        # 2. Map Images
        image_contexts = []
        img_id_to_file = {}
        for img in coco_data.get('images', []):
            img_id = str(img['id'])
            file_name = img.get('file_name', f"{img_id}.jpg")
            img_id_to_file[img_id] = file_name
            context = CSUImageContext(
                image_id=img_id,
                file_path=file_name,
                width=img.get('width', 0),
                height=img.get('height', 0),
                task_type="multi_task"
            )
            image_contexts.append(context)
            
        # 3. Map Annotations (Features)
        features = []
        for ann in coco_data.get('annotations', []):
            cat_id = ann.get('category_id', 0)
            cat_name = categories.get(cat_id, "Unknown")
            
            # COCO usually provides bbox: [x, y, width, height]
            # And segmentation: [[x1, y1, x2, y2, ...]]
            
            # Resolve object type
            obj_type = "annotation"
            geom_type = "BBox"
            coordinates = []
            
            if 'segmentation' in ann and len(ann['segmentation']) > 0 and isinstance(ann['segmentation'][0], list):
                obj_type = "cell" # We treat all segmentations as generic masks unless specified
                geom_type = "Polygon"
                
                # Convert COCO flattened [x1, y1, x2, y2] to [[x1, y1], [x2, y2]]
                flat_coords = ann['segmentation'][0]
                coords = []
                for i in range(0, len(flat_coords), 2):
                    if i + 1 < len(flat_coords):
                        coords.append([flat_coords[i], flat_coords[i+1]])
                # Close the polygon if not closed
                if coords and coords[0] != coords[-1]:
                    coords.append(coords[0])
                coordinates = [coords]
            elif 'bbox' in ann:
                obj_type = "detection"
                geom_type = "BBox"
                # COCO Bbox is [top_left_x, top_left_y, width, height] -> Represent natively or convert to corners
                x, y, w, h = ann['bbox']
                coordinates = [x, y, x + w, y + h]
            else:
                continue # Skip invalid
                
            geom = CSUGeometry(type=geom_type, coordinates=coordinates)
            props = CSUProperties(
                object_type=obj_type,
                classification=CSUClassProperties(name=cat_name, id=cat_id)
            )
            
            feat = CSUFeature(
                geometry=geom,
                properties=props,
                image_id=str(ann.get('image_id')),
                id=str(ann.get('id', ''))
            )
            features.append(feat)
            
        meta = CSUMetadata(generator="COCO_Converter_v1", extra_info=coco_data.get('info', {}))
        return CSUFeatureCollection(features=features, metadata=meta, images=image_contexts)

    def export_from_udf(self, udf_collection: CSUFeatureCollection, output_path: str, **kwargs) -> Any:
        # Boilerplate logic to convert UDF back to COCO dictionary
        # Needs to construct 'images', 'categories', 'annotations' arrays.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Simplified export implementation
        coco_dict = {
            "info": udf_collection.metadata.extra_info,
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Populate categories
        unique_cats = {}
        for feat in udf_collection.features:
            cls_prop = feat.properties.classification
            if cls_prop and cls_prop.id not in unique_cats:
                unique_cats[cls_prop.id] = cls_prop.name
                
        coco_dict["categories"] = [{"id": k, "name": v} for k, v in unique_cats.items()]
        
        # Populate images
        for img in udf_collection.images:
            coco_dict["images"].append({
                "id": img.image_id,
                "file_name": img.file_path,
                "width": img.width,
                "height": img.height
            })
            
        # Populate Annotations
        for i, feat in enumerate(udf_collection.features):
            cat_id = feat.properties.classification.id if feat.properties.classification else 0
            
            ann = {
                "id": feat.id if feat.id else str(i),
                "image_id": feat.image_id,
                "category_id": cat_id,
                "iscrowd": 0,
            }
            
            if feat.geometry.type == "Polygon":
                # Convert back to flattened coco format
                flat = []
                for point in feat.geometry.coordinates[0]:
                    flat.extend([point[0], point[1]])
                ann["segmentation"] = [flat]
                
                # Estimate bbox
                xs = [p[0] for p in feat.geometry.coordinates[0]]
                ys = [p[1] for p in feat.geometry.coordinates[0]]
                ann["bbox"] = [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]
                ann["area"] = ann["bbox"][2] * ann["bbox"][3] # dummy area
                
            elif feat.geometry.type == "BBox":
                # UDF Bbox: [xmin, ymin, xmax, ymax]
                c = feat.geometry.coordinates
                ann["bbox"] = [c[0], c[1], c[2]-c[0], c[3]-c[1]]
                ann["area"] = ann["bbox"][2] * ann["bbox"][3]
                
            coco_dict["annotations"].append(ann)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_dict, f, indent=2)
