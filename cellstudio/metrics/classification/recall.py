from sklearn.metrics import recall_score
from ..registry import MetricRegistry
from ..base import BaseMetric
@MetricRegistry.register("Recall")
class Recall(BaseMetric):
    def __init__(self, num_classes=2, **kwargs): self.avg = 'binary' if num_classes == 2 else 'macro'
    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:
        return float(recall_score(y_true, y_pred, average=self.avg, zero_division=0))
