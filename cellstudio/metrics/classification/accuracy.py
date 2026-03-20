from sklearn.metrics import accuracy_score
from ..registry import MetricRegistry
from ..base import BaseMetric
@MetricRegistry.register("Accuracy")
class Accuracy(BaseMetric):
    def __init__(self, **kwargs): pass
    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:
        return float(accuracy_score(y_true, y_pred))
