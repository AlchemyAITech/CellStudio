import numpy as np


def bbox_iou(box1, box2):
    if len(box2) == 0: return np.array([])
    x1 = np.maximum(box1[:, 0], box2[:, 0])
    y1 = np.maximum(box1[:, 1], box2[:, 1])
    x2 = np.minimum(box1[:, 2], box2[:, 2])
    y2 = np.minimum(box1[:, 3], box2[:, 3])
    inter_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter_area / (box1_area + box2_area - inter_area + 1e-16)

class DetMatchCache:
    _last_id = None
    _state = {}
    @classmethod
    def get(cls, true_boxes_list, pred_boxes_list, pred_scores_list=None, iou_thresh=0.5):
        flat_true = []
        flat_pred = []
        flat_scores = []
        
        if len(true_boxes_list) > 0 and isinstance(true_boxes_list[0], list):
            for b_idx in range(len(true_boxes_list)):
                batch_t = true_boxes_list[b_idx]
                batch_p = pred_boxes_list[b_idx]
                for i in range(len(batch_t)):
                    ds = batch_t[i]
                    if isinstance(batch_p, dict):
                        print(f"FATAL TYPING: pred_boxes_list[{b_idx}] is a dict! Keys: {batch_p.keys()}")
                    res = batch_p[i]
                    flat_true.append(ds.get('gt_bboxes'))
                    flat_pred.append(res.bboxes)
                    if hasattr(res, 'scores'):
                        flat_scores.append(res.scores)
                    else:
                        flat_scores.append(None)
            true_boxes_list = flat_true
            pred_boxes_list = flat_pred
            pred_scores_list = flat_scores
            
        current_id = id(true_boxes_list) ^ id(pred_boxes_list)
        if cls._last_id != current_id:
            cls._state = cls._compute(true_boxes_list, pred_boxes_list, pred_scores_list, iou_thresh)
            cls._last_id = current_id
        return cls._state

    @classmethod
    def _compute(cls, true_boxes_list, pred_boxes_list, pred_scores_list, iou_thresh):
        all_tp, all_fp, all_fn = 0, 0, 0
        iou_sum, match_count = 0.0, 0
        abs_count_errors, true_counts, pred_counts = [], [], []
        all_scores, all_matches = [], []
        for t_boxes, p_boxes, p_scores in zip(true_boxes_list, pred_boxes_list, pred_scores_list):
            t_boxes = np.array(t_boxes)
            p_boxes = np.array(p_boxes)
            p_scores = np.array(p_scores) if p_scores is not None else np.ones(len(p_boxes))
            true_counts.append(len(t_boxes))
            pred_counts.append(len(p_boxes))
            abs_count_errors.append(abs(len(t_boxes) - len(p_boxes)))
            if len(t_boxes) == 0 and len(p_boxes) == 0: continue
            if len(t_boxes) == 0:
                all_fp += len(p_boxes)
                all_scores.extend(p_scores.tolist())
                all_matches.extend([0]*len(p_boxes))
                continue
            if len(p_boxes) == 0:
                all_fn += len(t_boxes)
                continue
            sort_idx = np.argsort(p_scores)[::-1]
            p_boxes = p_boxes[sort_idx]
            p_scores = p_scores[sort_idx]
            matched_gt = set()
            for p_idx, p_box in enumerate(p_boxes):
                ious = bbox_iou(np.expand_dims(p_box, 0), t_boxes)
                if len(ious) == 0: continue
                best_gt_idx = np.argmax(ious)
                best_iou = ious[best_gt_idx]
                if best_iou >= iou_thresh and best_gt_idx not in matched_gt:
                    matched_gt.add(best_gt_idx)
                    all_tp += 1
                    iou_sum += best_iou
                    match_count += 1
                    all_scores.append(p_scores[p_idx])
                    all_matches.append(1)
                else:
                    all_fp += 1
                    all_scores.append(p_scores[p_idx])
                    all_matches.append(0)
            all_fn += len(t_boxes) - len(matched_gt)

        eps = 1e-8
        precision = all_tp / (all_tp + all_fp + eps)
        recall = all_tp / (all_tp + all_fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        ap50 = 0.0
        if len(all_scores) > 0:
            sort_scores_idx = np.argsort(all_scores)[::-1]
            matches_sorted = np.array(all_matches)[sort_scores_idx]
            tps = np.cumsum(matches_sorted)
            fps = np.cumsum(1 - matches_sorted)
            recalls = tps / (all_tp + all_fn + eps)
            precisions = tps / (tps + fps + eps)
            ap50 = np.trapz(precisions, recalls)
        return {"Precision": float(precision), "Recall": float(recall), "F1": float(f1), "mAP50": float(ap50),
                "Count_Error": float(np.mean(abs_count_errors)) if len(abs_count_errors)>0 else 0.0,
                "true_counts": true_counts, "pred_counts": pred_counts}
