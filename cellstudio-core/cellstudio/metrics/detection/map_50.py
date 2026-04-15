from ..base import BaseMetric
from ..registry import MetricRegistry
from .core import DetMatchCache


@MetricRegistry.register("det_map50")
class DetMAP50(BaseMetric):
    def __init__(self, iou_thresh=0.5, **kwargs): self.iou_thresh = iou_thresh
    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> dict:
        result = DetMatchCache.get(y_true, y_pred, y_prob, self.iou_thresh)
        # Return only scalar metrics (exclude list values like true_counts/pred_counts)
        return {k: v for k, v in result.items() if isinstance(v, (int, float))}
