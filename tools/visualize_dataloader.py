"""
Visual Probe for DataLoader Tensors vs UDF JSON Consistency
"""
import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cellstudio-core')))

def visualize_dataloader_batch(batch, save_dir="debug_output"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Typically batch is a dict combining batched arrays
    imgs = batch.get('img', [])
    bboxes = batch.get('gt_bboxes', [])
    labels = batch.get('gt_labels', [])

    if hasattr(imgs, 'numpy'):
        imgs = imgs.numpy()
        
    for i in range(len(imgs)):
        img = imgs[i]
        # De-normalize if normalized (Assuming basic standard mean/std or simple 0-255 uint8)
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0)
            img = img.astype(np.uint8)
            
        # Ensure HWC
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
            
        # Draw BBoxes
        if i < len(bboxes):
            bxs = bboxes[i]
            if hasattr(bxs, 'numpy'):
                bxs = bxs.numpy()
            for box in bxs:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        # Save Probe Image
        save_path = os.path.join(save_dir, f"probe_batch_item_{i}.png")
        # Need to flip RGB to BGR for cv2
        if img.shape[-1] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        cv2.imwrite(save_path, img_bgr)
        print(f"Visualized batch item saved to: {save_path}")

if __name__ == '__main__':
    print("Dataloader Vision Probe Ready. Hook into your pipeline's training loop to dump batches.")
