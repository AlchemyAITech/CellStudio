import numpy as np
from typing import List, Tuple, Dict

class WSIProcessor:
    """
    Pathology Gigapixel Image (Whole Slide Image) Processor.
    Handles memory-safe sliding window cropping and Soft-NMS tile fusion.
    """
    
    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = self.tile_size - self.overlap
        
    def generate_tiles(self, image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Generates bounding boxes for tiles covering the entire WSI.
        Returns: [{"box": [x1, y1, x2, y2], "id": idx}, ...]
        """
        h_img, w_img = image_shape[:2]
        tiles = []
        idx = 0
        
        for y in range(0, h_img, self.stride):
            for x in range(0, w_img, self.stride):
                x2 = min(x + self.tile_size, w_img)
                y2 = min(y + self.tile_size, h_img)
                
                # Adjust if chunk is too small at the edges
                if x2 - x < self.tile_size and x - self.stride >= 0:
                    x = max(0, x2 - self.tile_size)
                if y2 - y < self.tile_size and y - self.stride >= 0:
                    y = max(0, y2 - self.tile_size)
                    
                tiles.append({
                    "id": idx,
                    "box": [x, y, x2, y2]
                })
                idx += 1
                
        return tiles

    @staticmethod
    def soft_nms(boxes: np.ndarray, scores: np.ndarray, sigma: float = 0.5, Nt: float = 0.3, threshold: float = 0.001, method: int = 1):
        """
        Soft Non-Maximum Suppression to fuse edge-overlapping detection boxes across WSI tiles.
        method: 1 (Linear), 2 (Gaussian)
        """
        N = boxes.shape[0]
        indexes = np.arange(N)
        
        for i in range(N):
            # Find the max score among remaining boxes
            maxscore = scores[i]
            maxpos = i
            
            for index in range(i + 1, N):
                if scores[index] > maxscore:
                    maxscore = scores[index]
                    maxpos = index
                    
            # Swap max to i'th position
            boxes[[i, maxpos]] = boxes[[maxpos, i]]
            scores[[i, maxpos]] = scores[[maxpos, i]]
            indexes[[i, maxpos]] = indexes[[maxpos, i]]
            
            # Apply decay to overlapping boxes
            x1 = boxes[i, 0]
            y1 = boxes[i, 1]
            x2 = boxes[i, 2]
            y2 = boxes[i, 3]
            area_i = (x2 - x1 + 1) * (y2 - y1 + 1)
            
            for pos in range(i + 1, N):
                xx1 = np.maximum(x1, boxes[pos, 0])
                yy1 = np.maximum(y1, boxes[pos, 1])
                xx2 = np.minimum(x2, boxes[pos, 2])
                yy2 = np.minimum(y2, boxes[pos, 3])
                
                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                area_pos = (boxes[pos, 2] - boxes[pos, 0] + 1) * (boxes[pos, 3] - boxes[pos, 1] + 1)
                iou = inter / (area_i + area_pos - inter)
                
                if method == 1:  # Linear
                    if iou > Nt:
                        weight = 1 - iou
                    else:
                        weight = 1
                elif method == 2:  # Gaussian
                    weight = np.exp(-(iou * iou) / sigma)
                else:  # Original NMS
                    if iou > Nt:
                        weight = 0
                    else:
                        weight = 1
                        
                scores[pos] = scores[pos] * weight
                
        keep = indexes[scores > threshold]
        return keep

    def reconstruct_wsi_detections(self, tile_predictions: List[Dict], tile_metadata: List[Dict]) -> Dict:
        """
        Takes predictions from each tile, maps them back to global WSI coordinates, 
        and applies Soft-NMS to remove duplicate overlapping edge predictions.
        """
        global_boxes = []
        global_scores = []
        global_labels = []
        
        for pred, meta in zip(tile_predictions, tile_metadata):
            tx, ty, _, _ = meta["box"]
            for box, score, label in zip(pred.get("boxes", []), pred.get("scores", []), pred.get("labels", [])):
                # Shift local coordinates to global
                g_box = [box[0] + tx, box[1] + ty, box[2] + tx, box[3] + ty]
                global_boxes.append(g_box)
                global_scores.append(score)
                global_labels.append(label)
                
        if not global_boxes:
            return {"boxes": [], "scores": [], "labels": []}
            
        g_np_boxes = np.array(global_boxes, dtype=np.float32)
        g_np_scores = np.array(global_scores, dtype=np.float32)
        g_np_labels = np.array(global_labels, dtype=np.int32)
        
        # We need to run Soft-NMS per class independently
        unique_labels = np.unique(g_np_labels)
        final_keep_indices = []
        
        for lbl in unique_labels:
            class_mask = g_np_labels == lbl
            c_boxes = g_np_boxes[class_mask]
            c_scores = g_np_scores[class_mask]
            
            keep = self.soft_nms(c_boxes, c_scores, method=2)  # Gaussian Soft-NMS
            
            # Map back to original indices
            original_indices = np.where(class_mask)[0]
            final_keep_indices.extend(original_indices[keep].tolist())
            
        final_boxes = g_np_boxes[final_keep_indices].tolist()
        final_scores = g_np_scores[final_keep_indices].tolist()
        final_labels = g_np_labels[final_keep_indices].tolist()
        
        return {
            "boxes": final_boxes,
            "scores": final_scores,
            "labels": final_labels
        }
