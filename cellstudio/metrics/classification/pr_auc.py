from sklearn.metrics import average_precision_score
from ..registry import MetricRegistry
from ..base import BaseMetric
@MetricRegistry.register("PR_AUC")
class PRAUC(BaseMetric):
    def __init__(self, num_classes=2, **kwargs): self.num_classes = num_classes
    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:
        if y_prob is None or self.num_classes != 2: return 0.0
        prob = y_prob if y_prob.ndim == 1 or y_prob.shape[1] == 1 else y_prob[:, 1]
        return float(average_precision_score(y_true, prob))
