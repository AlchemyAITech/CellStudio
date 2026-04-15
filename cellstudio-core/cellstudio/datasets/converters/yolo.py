import os
from typing import Dict, Any, List

from .base import BaseConverter
from ...structures.csuos import (
    CSUFeatureCollection, CSUFeature, CSUProperties, CSUGeometry, 
    CSUClassProperties, CSUImageContext, CSUMetadata
)

class YOLOConverter(BaseConverter):
    """Parses standard YOLO format (darknet .txt) into UDF format.
    
    YOLO parses folder structure rather than a single JSON file.
    Assumes matching images and `.txt` files with `image_id`.
    """
    
    def parse_to_udf(self, source_path: str, labels_dir: str, class_names: List[str], **kwargs) -> CSUFeatureCollection:
        """Parse YOLO directory.
        
        Args:
            source_path: Path to images directory.
            labels_dir: Path to annotations directory.
            class_names: List of class names where index matches YOLO id.
        """
        features = []
        image_contexts = []
        
        from PIL import Image # For getting image dimensions
        
        for img_name in os.listdir(source_path):
            if not img_name.lower().endswith(('.jpg', '.png', '.tif', '.jpeg')):
                continue
                
            img_id = os.path.splitext(img_name)[0]
            img_path = os.path.join(source_path, img_name)
            txt_path = os.path.join(labels_dir, f"{img_id}.txt")
            
            # Read image metadata
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                w, h = 1000, 1000 # Dummy fallback
                
            context = CSUImageContext(
                image_id=img_id,
                file_path=img_path,
                width=w,
                height=h,
                task_type="detection"
            )
            image_contexts.append(context)
            
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line_idx, line in enumerate(f.readlines()):
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        
                        class_id = int(parts[0])
                        c_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
                        
                        # YOLO format: class x_center y_center width height (Normalized 0.0 - 1.0)
                        x_c, y_c, bw, bh = map(float, parts[1:5])
                        
                        # Denormalize to absolute pixels for UDF
                        abs_x_c = x_c * w
                        abs_y_c = y_c * h
                        abs_w = bw * w
                        abs_h = bh * h
                        
                        xmin = abs_x_c - (abs_w / 2)
                        ymin = abs_y_c - (abs_h / 2)
                        xmax = abs_x_c + (abs_w / 2)
                        ymax = abs_y_c + (abs_h / 2)
                        
                        geom = CSUGeometry(type="BBox", coordinates=[xmin, ymin, xmax, ymax])
                        props = CSUProperties(
                            object_type="detection",
                            classification=CSUClassProperties(name=c_name, id=class_id)
                        )
                        
                        feat = CSUFeature(
                            geometry=geom,
                            properties=props,
                            image_id=img_id,
                            id=f"{img_id}_{line_idx}"
                        )
                        features.append(feat)
                        
        meta = CSUMetadata(generator="YOLO_Converter_v1")
        return CSUFeatureCollection(features=features, metadata=meta, images=image_contexts)

    def export_from_udf(self, udf_collection: CSUFeatureCollection, output_dir: str, **kwargs) -> Any:
        # Export logic back to TXT files
        os.makedirs(output_dir, exist_ok=True)
        img_dict = {img.image_id: img for img in udf_collection.images}
        
        # Group features by image
        img_features = {}
        for feat in udf_collection.features:
            if feat.image_id not in img_features:
                img_features[feat.image_id] = []
            img_features[feat.image_id].append(feat)
            
        for img_id, feats in img_features.items():
            context = img_dict.get(img_id)
            if not context or context.width == 0:
                continue # Cannot normalize without width
                
            w = context.width
            h = context.height
            
            lines = []
            for feat in feats:
                if feat.geometry.type != "BBox":
                    continue # Ignore segmentations for standard 5-part YOLO
                c = feat.geometry.coordinates
                xmin, ymin, xmax, ymax = c[0], c[1], c[2], c[3]
                
                # Normalize
                x_c = ((xmin + xmax) / 2) / w
                y_c = ((ymin + ymax) / 2) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                
                cat_id = feat.properties.classification.id if feat.properties.classification else 0
                lines.append(f"{cat_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
                
            if lines:
                with open(os.path.join(output_dir, f"{img_id}.txt"), 'w') as f:
                    f.writelines(lines)
