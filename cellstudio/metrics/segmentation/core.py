import numpy as np
import cv2
from scipy.spatial.distance import cdist

def compute_instance_iou(gt_mask_binary, pred_mask_binary):
    intersection = np.logical_and(gt_mask_binary, pred_mask_binary).sum()
    union = np.logical_or(gt_mask_binary, pred_mask_binary).sum()
    return intersection / (union + 1e-6), intersection, union

def compute_hd95(gt_mask, pred_mask):
    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not gt_contours and not pred_contours: return 0.0
    if not gt_contours or not pred_contours: return 100.0
    gt_pts = np.vstack([c.squeeze() for c in gt_contours if c.shape[0] > 0])
    pred_pts = np.vstack([c.squeeze() for c in pred_contours if c.shape[0] > 0])
    if gt_pts.ndim == 1: gt_pts = np.expand_dims(gt_pts, 0)
    if pred_pts.ndim == 1: pred_pts = np.expand_dims(pred_pts, 0)
    dists = cdist(gt_pts, pred_pts)
    return max(np.percentile(np.min(dists, axis=1), 95), np.percentile(np.min(dists, axis=0), 95))

class SegMatchCache:
    _last_id = None
    _state = {}

    @classmethod
    def get(cls, true_masks_list, pred_masks_list, iou_thresh=0.5):
        current_id = id(true_masks_list) ^ id(pred_masks_list)
        if cls._last_id != current_id:
            cls._state = cls._aggregate(true_masks_list, pred_masks_list, iou_thresh)
            cls._last_id = current_id
        return cls._state

    @classmethod
    def _aggregate(cls, true_masks_list, pred_masks_list, iou_thresh):
        metrics = {"Dice": [], "mIoU": [], "PQ": [], "AJI": [], "HD95": []}
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
        matched_preds = set()
        gt_masks = {i: (gt_map == i) for i in gt_ids}
        pred_masks = {j: (pred_map == j) for j in pred_ids}
        for gt_id in gt_ids:
            gt_m = gt_masks[gt_id]
            best_iou, best_p, best_inter, best_union = 0.0, -1, 0, 0
            for p_id in pred_ids:
                if p_id in matched_preds: continue
                iou, inter, un = compute_instance_iou(gt_m, pred_masks[p_id])
                if iou > best_iou: best_iou, best_p, best_inter, best_union = iou, p_id, inter, un
            if best_iou >= iou_thresh:
                tp += 1; fn -= 1; fp -= 1
                iou_sum += best_iou
                matched_preds.add(best_p)
                aji_inter += best_inter
                aji_union += best_union
            else: aji_union += gt_m.sum()
        for p_id in pred_ids:
            if p_id not in matched_preds: aji_union += pred_masks[p_id].sum()
        sq = iou_sum / (tp + 1e-8)
        rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)
        return {"Dice": dice, "mIoU": miou, "PQ": sq * rq, "AJI": aji_inter / (aji_union + 1e-8), "HD95": compute_hd95(gt_bin, pred_bin)}
