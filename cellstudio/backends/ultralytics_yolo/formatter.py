import os
import yaml
from pathlib import Path
from typing import Dict
from pathstudio.datasets.schema import CellDatasetConfig

class YoloDataFormatter:
    """
    Translates PathStudio JSON Schema standard dataset into Ultralytics YOLO temporary format.
    YOLO requires images in an images/ folder and corresponding .txt labels in labels/.
    This isolates YOLO-specific formats from the rest of the generic architecture.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        
    def format_from_json(self, json_path: str, task_type: str = "detect") -> str:
        """
        Takes a PathStudio JSON file and writes out YOLO-compliant symlinks/copies and data.yaml.
        Returns path to the generated data.yaml.
        """
        config = CellDatasetConfig.load(json_path)
        
        # Create YOLO structure
        base_dir = os.path.join(self.output_dir, "yolo_formatted")
        images_dir = os.path.join(base_dir, "images", "train")
        labels_dir = os.path.join(base_dir, "labels", "train")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        class_map = {name: i for i, name in enumerate(config.classes)}
        
        for item in config.items:
            # Force target to be JPG and convert to 3 channels to prevent YOLO TIFF 4-channel loader crashes
            base_name = os.path.splitext(os.path.basename(item.image_path))[0]
            img_basename = f"{base_name}.jpg"
            target_img = os.path.join(images_dir, img_basename)
            
            if not os.path.lexists(target_img):
                try:
                    from PIL import Image
                    with Image.open(item.image_path) as img:
                        img_rgb = img.convert('RGB')
                        img_rgb.save(target_img, 'JPEG')
                except Exception as e:
                    print(f"[YoloDataFormatter] Warning: could not convert {item.image_path}: {e}")
                    
            # Write YOLO label format
            label_basename = f"{base_name}.txt"
            target_label = os.path.join(labels_dir, label_basename)
            
            with open(target_label, 'w') as f:
                if task_type in ["detect", "detection"]:
                    for bbox in item.bboxes:
                        # YOLO format: class x_center y_center width height (normalized)
                        # We must compute normalized if image size is known
                        iw = item.image_width
                        ih = item.image_height
                        
                        if iw and ih:
                            dw = 1.0 / iw
                            dh = 1.0 / ih
                            
                            w = (bbox.xmax - bbox.xmin)
                            h = (bbox.ymax - bbox.ymin)
                            x = bbox.xmin + (w / 2.0)
                            y = bbox.ymin + (h / 2.0)
                            
                            xc = x * dw
                            yc = y * dh
                            wn = w * dw
                            hn = h * dh
                            
                            cid = class_map.get(bbox.label, 0)
                            f.write(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
                        else:
                            # If size unknown, assume annotations are already scaled or skip safely 
                            pass 

                elif task_type in ["segment", "segmentation"]:
                    for poly in item.polygons:
                        cid = class_map.get(poly.label, 0)
                        
                        iw = item.image_width
                        ih = item.image_height
                        
                        if iw and ih and len(poly.points) % 2 == 0:
                            dw = 1.0 / iw
                            dh = 1.0 / ih
                            
                            norm_pts = []
                            for i in range(0, len(poly.points), 2):
                                px = poly.points[i] * dw
                                py = poly.points[i+1] * dh
                                norm_pts.extend([f"{px:.6f}", f"{py:.6f}"])
                                
                            f.write(f"{cid} {' '.join(norm_pts)}\n")
        
        # Write yaml
        yaml_content = {
            "path": os.path.abspath(base_dir),
            "train": "images/train",
            "val": "images/train", # for MVP use train as val
            "names": {i: name for i, name in enumerate(config.classes)}
        }
        
        yaml_path = os.path.join(base_dir, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
            
        return yaml_path
