from ..base import BaseMetric
from ..registry import MetricRegistry
from .core import SegMatchCache


@MetricRegistry.register("seg_all_metrics")
class SegAllMetrics(BaseMetric):
    def __init__(self, iou_thresh=0.5, **kwargs): 
        self.iou_thresh = iou_thresh
        
    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> dict:
        # Returns Dict: {"Dice": ..., "mIoU": ..., "PQ": ..., "AJI": ..., "HD95": ..., "Recall": ..., "Precision": ..., "F1": ..., "Count_MAE": ...}
        return SegMatchCache.get(y_true, y_pred, self.iou_thresh)
