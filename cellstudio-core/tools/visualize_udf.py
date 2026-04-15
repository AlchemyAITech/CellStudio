import os
import json
import argparse
import numpy as np
import cv2
from colorsys import hsv_to_rgb

def get_color(class_id, max_classes=80):
    """Generate a distinct bright color mapping based on ID"""
    hue = (class_id * 137.508) % 360 / 360.0
    r, g, b = hsv_to_rgb(hue, 0.8, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255)) # BGR for OpenCV

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CellStudio UDF Format')
    parser.add_argument('json_path', help='Path to udf_standard.json or udf_predictions.json')
    parser.add_argument('--img-dir', help='Path to original images directory (Optional, attempts to resolve from JSON)')
    parser.add_argument('--out-dir', default='outputs/visualizations', help='Directory to save rendered images')
    parser.add_argument('--limit', type=int, default=10, help='Maximum images to process')
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    images = data.get('images', [])
    features = data.get('features', [])
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Map features to images
    img_feature_map = {img['image_id']: [] for img in images}
    for feat in features:
        iid = feat.get('image_id')
        if iid in img_feature_map:
            img_feature_map[iid].append(feat)
            
    print(f"Loaded {len(images)} images and {len(features)} features from UDF JSON.")
    
    count = 0
    for img_meta in images:
        if count >= args.limit:
            break
            
        img_id = img_meta['image_id']
        file_path = img_meta.get('file_path', '')
        
        # 1. Resolve Image Path
        target_path = file_path
        if args.img_dir:
            target_path = os.path.join(args.img_dir, os.path.basename(file_path))
        else:
            # Attempt to resolve heuristically from json path location
            json_dir = os.path.dirname(os.path.abspath(args.json_path))
            target_path = os.path.join(json_dir, file_path)
            if not os.path.exists(target_path):
                target_path = os.path.join(json_dir, os.path.basename(file_path))
                if not os.path.exists(target_path):
                    target_path = os.path.join(json_dir, 'images', os.path.basename(file_path))
                
        if not os.path.exists(target_path):
            print(f"[Warning] Could not locate image {target_path}, skipping visualization.")
            continue
            
        print(f"Rendering {os.path.basename(target_path)} ...")
        
        # 2. Read Image
        img = cv2.imread(target_path)
        if img is None:
            continue
            
        # 3. Canvas for alpha blending
        overlay = img.copy()
        
        # 4. Draw Features
        for feat in img_feature_map.get(img_id, []):
            geom = feat.get('geometry', {})
            props = feat.get('properties', {})
            cls_prop = props.get('class', {})
            
            cat_name = cls_prop.get('name', 'unknown')
            # Extract id deterministically or hash name for color
            cat_id = cls_prop.get('id', abs(hash(cat_name)) % 80)
            color = get_color(cat_id)
            
            geom_type = geom.get('type')
            coords = geom.get('coordinates', [])
            
            if geom_type == 'BBox' and len(coords) == 4:
                # [x, y, w, h] if coco, [x1, y1, x2, y2] if CellStudio Standard
                # Cellstudio natively standardizes on bounding boxes [x1, y1, x2, y2] in geometries (polygon-bounds)
                x1, y1, x2, y2 = [int(v) for v in coords]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, int(img.shape[1]/500)))
                cv2.putText(img, cat_name, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
            elif geom_type == 'Polygon' and len(coords) > 0:
                # Polygons usually nested [ [[x,y], [x,y]] ]
                try:
                    poly_arr = np.array(coords[0], dtype=np.int32)
                    cv2.fillPoly(overlay, [poly_arr], color)
                    # draw border
                    cv2.polylines(img, [poly_arr], isClosed=True, color=color, thickness=max(1, int(img.shape[1]/1000)))
                    
                    if len(poly_arr) > 0:
                        cx, cy = int(np.mean(poly_arr[:, 0])), int(np.mean(poly_arr[:, 1]))
                        cv2.putText(img, cat_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                except Exception as e:
                    pass
        
        # Alpha blend polygons
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # 5. Global Classifier Overlays
        global_labels = set()
        for feat in img_feature_map.get(img_id, []):
            if feat.get('properties', {}).get('object_type') == 'image':
                global_labels.add(feat.get('properties', {}).get('class', {}).get('name', 'unknown'))
                
        if global_labels:
            msg = f"Task[Classification]: {', '.join(global_labels)}"
            cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0,0,0), -1)
            cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        # Save output
        out_path = os.path.join(args.out_dir, f"render_{os.path.basename(target_path)}")
        cv2.imwrite(out_path, img)
        count += 1

    print(f"Success! {count} Visualizations written to {args.out_dir}/")

if __name__ == '__main__':
    main()
