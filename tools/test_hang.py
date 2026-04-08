import sys  # noqa: F401
import os  # noqa: F401
import cv2
import json
import numpy as np
from cellpose import dynamics
import time

ann_file = 'datasets/segmentation/cellpose/splits/train_fold0.json'
with open(ann_file, 'r') as f:
    data = json.load(f)

for img_info in data['images'][:3]:
    img_id = img_info['id']
    polygons = [ann['segmentation'][0] for ann in data['annotations'] if ann['image_id'] == img_id]
    
    h, w = img_info['height'], img_info['width']
    gt_instance_seg = np.zeros((h, w), dtype=np.int32)
    for i, poly in enumerate(polygons):
        pts = np.array(poly).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(gt_instance_seg, [pts], i + 1)
        
    # Resize heavily with NEAREST like the pipeline
    resized_mask = cv2.resize(gt_instance_seg, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    print(f"Testing flow computation on Image {img_id} with {len(polygons)} instances...")
    t0 = time.time()
    out = dynamics.labels_to_flows([resized_mask])[0]
    t1 = time.time()
    print(f"Finished in {t1-t0:.4f}s")
