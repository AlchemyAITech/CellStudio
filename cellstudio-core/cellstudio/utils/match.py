import numpy as np
from shapely.geometry import Polygon
from cellstudio.structures.csuos import CSUFeature, CSUFeatureCollection

def feat_to_polygon(feat: CSUFeature) -> Polygon:
    t = feat.geometry.type
    coords = feat.geometry.coordinates
    if t == 'BBox':
        xmi, ymi, xma, yma = coords[0], coords[1], coords[2], coords[3]
        return Polygon([(xmi, ymi), (xma, ymi), (xma, yma), (xmi, yma)])
    elif t == 'Polygon':
        # Single exterior ring for simplistic calculation
        ring = coords[0]
        if len(ring) < 3:
            return Polygon()
        return Polygon(ring)
    return Polygon()

def compute_udf_iou(feat1: CSUFeature, feat2: CSUFeature) -> float:
    """Compute IoU between two UDF geometries."""
    try:
        p1 = feat_to_polygon(feat1)
        p2 = feat_to_polygon(feat2)
        if not p1.is_valid or not p2.is_valid or p1.is_empty or p2.is_empty:
            return 0.0
        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        if union == 0:
            return 0.0
        return inter / union
    except Exception:
        return 0.0

class MatchCache:
    """Associates Algorithm Predictions with Human GT based on physical IoU thresholding."""
    
    def __init__(self, iou_thresh: float = 0.4):
        self.iou_thresh = iou_thresh

    def match_collections(self, gt_coll: CSUFeatureCollection, pred_coll: CSUFeatureCollection):
        """
        Modifies pred_coll features in-place by injecting `match_info`.
        Using a simple greedy matching algorithm to find best hits above the threshold.
        """
        gts = gt_coll.features
        preds = pred_coll.features
        
        matched_gt_ids = set()
        
        # O(N^2) greedy matching
        for p in preds:
            best_iou = 0.0
            best_gt = None
            
            for gt in gts:
                if gt.id in matched_gt_ids:
                    continue
                iou = compute_udf_iou(p, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
            
            if best_iou >= self.iou_thresh and best_gt is not None:
                p.properties.match_info = {
                    'gt_id': best_gt.id,
                    'iou': round(float(best_iou), 4)
                }
                matched_gt_ids.add(best_gt.id)
