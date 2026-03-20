from sklearn.metrics import precision_score
from ..registry import MetricRegistry
from ..base import BaseMetric
@MetricRegistry.register("Precision")
class Precision(BaseMetric):
    def __init__(self, num_classes=2, **kwargs): self.avg = 'binary' if num_classes == 2 else 'macro'
    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:
        return float(precision_score(y_true, y_pred, average=self.avg, zero_division=0))
