from sklearn.metrics import cohen_kappa_score

from ..base import BaseMetric
from ..registry import MetricRegistry


@MetricRegistry.register("Kappa")
class Kappa(BaseMetric):
    def __init__(self, **kwargs): pass
    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:
        return float(cohen_kappa_score(y_true, y_pred))
