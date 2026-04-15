from ..base import BaseMetric
from ..registry import MetricRegistry
from .core import DetMatchCache


@MetricRegistry.register("det_precision")
class DetPrecision(BaseMetric):
    def __init__(self, iou_thresh=0.5, **kwargs): self.iou_thresh = iou_thresh
    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:
        return DetMatchCache.get(y_true, y_pred, y_prob, self.iou_thresh)["Precision"]
