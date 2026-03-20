from sklearn.metrics import f1_score
from ..registry import MetricRegistry
from ..base import BaseMetric
@MetricRegistry.register("F1_Score")
class F1Score(BaseMetric):
    def __init__(self, num_classes=2, **kwargs): self.avg = 'binary' if num_classes == 2 else 'macro'
    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:
        return float(f1_score(y_true, y_pred, average=self.avg, zero_division=0))
