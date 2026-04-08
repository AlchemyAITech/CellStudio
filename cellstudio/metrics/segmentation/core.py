import numpy as np
import torch


def compute_instance_iou(gt_mask_binary, pred_mask_binary):
    intersection = np.logical_and(gt_mask_binary, pred_mask_binary).sum()
    union = np.logical_or(gt_mask_binary, pred_mask_binary).sum()
    return intersection / (union + 1e-6), intersection, union

def compute_hd95(gt_mask, pred_mask):
    # Extract edges using morphological operations or just subtract erosion
    import cv2
    from scipy.ndimage import distance_transform_edt
    kernel = np.ones((3,3), np.uint8)
    gt_erode = cv2.erode(gt_mask.astype(np.uint8), kernel)
    gt_edges = (gt_mask > 0) & (gt_erode == 0)
    
    pred_erode = cv2.erode(pred_mask.astype(np.uint8), kernel)
    pred_edges = (pred_mask > 0) & (pred_erode == 0)
    
    if not np.any(gt_edges) and not np.any(pred_edges): return 0.0
    if not np.any(gt_edges) or not np.any(pred_edges): return 100.0
    
    # Distance transform
    dt_gt = distance_transform_edt(~gt_edges)
    dt_pred = distance_transform_edt(~pred_edges)
    
    # HD95 is the 95th percentile of the distances
    dists_a_to_b = dt_pred[gt_edges]
    dists_b_to_a = dt_gt[pred_edges]
    
    # Get 95th percentile of each directed distance, then take the max
    return float(max(np.percentile(dists_a_to_b, 95), np.percentile(dists_b_to_a, 95)))

class SegMatchCache:
    _last_id = None
    _state = {}

    @classmethod
    def get(cls, true_masks_list, pred_masks_list, iou_thresh=0.5):
        # 1. Unpack hierarchical DataSamples and InferResults if they match CellStudio signatures
        flat_gt = []
        flat_pred = []
        
        # Unpack GT
        if len(true_masks_list) > 0 and isinstance(true_masks_list[0], list):
            for batch in true_masks_list:
                for sample in batch:
                    # Look for explicit gt_instance_seg (H, W numpy array or tensor)
                    if hasattr(sample, 'gt_instance_seg'):
                        val = sample.gt_instance_seg
                        if torch.is_tensor(val): val = val.cpu().numpy()
                        if val.ndim == 3:
                            inst_map = np.zeros(val.shape[1:], dtype=np.int32)
                            for i in range(val.shape[0]):
                                inst_map[val[i] > 0] = i + 1
                            flat_gt.append(inst_map)
                        else:
                            flat_gt.append(val)
                    elif hasattr(sample, 'gt_instances') and sample.gt_instances is not None and sample.gt_instances.masks is not None:
                        masks = sample.gt_instances.masks
                        if torch.is_tensor(masks): masks = masks.cpu().numpy()
                        # Convert [N, H, W] boolean/binary layers into [H, W] instance ID map
                        inst_map = np.zeros(masks.shape[1:], dtype=np.int32)
                        for i in range(masks.shape[0]):
                            inst_map[masks[i] > 0] = i + 1
                        flat_gt.append(inst_map)
                    else:
                        flat_gt.append(None)
                        
        else:
            flat_gt = true_masks_list
            
        # Unpack Pred
        if len(pred_masks_list) > 0 and isinstance(pred_masks_list[0], list):
            for batch in pred_masks_list:
                for res in batch:
                    if hasattr(res, 'masks') and res.masks is not None:
                        masks = res.masks
                        if torch.is_tensor(masks): masks = masks.cpu().numpy()
                        
                        if len(masks) == 0:
                            if len(flat_gt) > len(flat_pred) and flat_gt[len(flat_pred)] is not None:
                                h, w = flat_gt[len(flat_pred)].shape
                                flat_pred.append(np.zeros((h, w), dtype=np.int32))
                            else:
                                flat_pred.append(np.zeros((256, 256), dtype=np.int32)) # fallback
                            continue
                            
                        # Convert [N, H, W] into instance ID map
                        inst_map = np.zeros(masks.shape[1:], dtype=np.int32)
                        for i in range(masks.shape[0]):
                            inst_map[masks[i] > 0] = i + 1
                        flat_pred.append(inst_map)
                    else:
                        flat_pred.append(None)
        else:
            flat_pred = pred_masks_list
            
        # Filter None
        valid_pairs = [(g, p) for g, p in zip(flat_gt, flat_pred) if g is not None and p is not None]
        if not valid_pairs:
            return {"Dice": 0.0, "mIoU": 0.0, "PQ": 0.0, "AJI": 0.0, "HD95": 0.0, "Recall": 0.0, "Precision": 0.0, "F1": 0.0}
            
        final_gt = [vp[0] for vp in valid_pairs]
        final_pred = [vp[1] for vp in valid_pairs]
                
        current_id = id(true_masks_list) ^ id(pred_masks_list)
        if cls._last_id != current_id:
            cls._state = cls._aggregate(final_gt, final_pred, iou_thresh)
            cls._last_id = current_id
        return cls._state

    @classmethod
    def _aggregate(cls, true_masks_list, pred_masks_list, iou_thresh):
        metrics = {"Dice": [], "mIoU": [], "PQ": [], "AJI": [], "HD95": [], "Recall": [], "Precision": [], "F1": [], "Count_MAE": []}
        for gt_map, pred_map in zip(true_masks_list, pred_masks_list):
            res = cls._compute_single(gt_map, pred_map, iou_thresh)
            for k in metrics: metrics[k].append(res[k])
        return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}

    @classmethod
    def _compute_single(cls, gt_map, pred_map, iou_thresh):
        gt_ids = [i for i in np.unique(gt_map) if i != 0]
        pred_ids = [j for j in np.unique(pred_map) if j != 0]
        gt_bin = (gt_map > 0)
        pred_bin = (pred_map > 0)
        
        intersect = np.logical_and(gt_bin, pred_bin).sum()
        union = np.logical_or(gt_bin, pred_bin).sum()
        dice = 2.0 * intersect / (gt_bin.sum() + pred_bin.sum() + 1e-8)
        miou = intersect / (union + 1e-8)
        
        tp, fp, fn = 0, len(pred_ids), len(gt_ids)
        iou_sum, aji_inter, aji_union = 0.0, 0.0, 0.0
        
        max_gt = max(gt_ids) if gt_ids else 0
        max_pred = max(pred_ids) if pred_ids else 0
        
        if max_gt > 0 and max_pred > 0:
            hist_size = (max_gt + 1) * (max_pred + 1)
            hist = np.bincount(
                gt_map.ravel() * (max_pred + 1) + pred_map.ravel(),
                minlength=hist_size
            ).reshape(max_gt + 1, max_pred + 1)
        else:
            hist = np.zeros((max_gt + 1, max_pred + 1), dtype=np.int32)
            
        gt_areas = np.bincount(gt_map.ravel(), minlength=max_gt + 1)
        pred_areas = np.bincount(pred_map.ravel(), minlength=max_pred + 1)
        
        matched_preds = set()
        
        for gt_id in gt_ids:
            best_iou = 0.0
            best_p = -1
            best_inter = 0
            best_un = 0
            
            for p_id in pred_ids:
                if p_id in matched_preds:
                    continue
                
                inter = hist[gt_id, p_id]
                if inter == 0:
                    continue
                    
                un = gt_areas[gt_id] + pred_areas[p_id] - inter
                iou = inter / (un + 1e-8)
                
                if iou > best_iou:
                    best_iou = iou
                    best_p = p_id
                    best_inter = inter
                    best_un = un
                    
            if best_iou >= iou_thresh:
                tp += 1
                fn -= 1
                fp -= 1
                iou_sum += best_iou
                matched_preds.add(best_p)
                aji_inter += best_inter
                aji_union += best_un
            else:
                aji_union += gt_areas[gt_id]
                
        for p_id in pred_ids:
            if p_id not in matched_preds:
                aji_union += pred_areas[p_id]
                
        sq = iou_sum / (tp + 1e-8)
        rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)
        
        recall = tp / (len(gt_ids) + 1e-8)
        precision = tp / (len(pred_ids) + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            "Dice": dice, 
            "mIoU": miou, 
            "PQ": sq * rq, 
            "AJI": aji_inter / (aji_union + 1e-8), 
            "HD95": compute_hd95(gt_bin, pred_bin),
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "Count_MAE": abs(len(pred_ids) - len(gt_ids))
        }
