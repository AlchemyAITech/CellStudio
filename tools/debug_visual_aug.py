"""
Data Augmentation Visual Targeting Engine
"""
import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cellstudio-core')))
from cellstudio.pipeline.transforms.visual_aug import PathologyRotateExpand

def debug_pathology_rotate():
    """Generates a dummy image with known bounding boxes to simulate margin-rotation"""
    print("Testing PathologyRotateExpand ...")
    
    # Create 800x800 image simulating a chunk retrieved WITH margin
    img = np.ones((800, 800, 3), dtype=np.uint8) * 128
    
    # Draw a mock tumor cell at the absolute center
    cv2.circle(img, (400, 400), 100, (0, 0, 255), -1)
    
    # Single Bbox around it
    bbox = np.array([[300, 300, 500, 500]], dtype=np.float32)
    
    # Pack result
    results = {
        'img': img,
        'gt_bboxes': bbox,
        'gt_labels': np.array([1])
    }
    
    transform = PathologyRotateExpand(angles=[45], prob=1.0) # force 45 degree rot
    out = transform(results)
    
    out_img = out['img']
    out_boxes = out['gt_bboxes']
    
    for b in out_boxes:
        x1, y1, x2, y2 = map(int, b[:4])
        cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
    os.makedirs("debug_output", exist_ok=True)
    cv2.imwrite("debug_output/aug_rotated.png", out_img)
    print("Augmentation preview saved to debug_output/aug_rotated.png")

if __name__ == "__main__":
    debug_pathology_rotate()
