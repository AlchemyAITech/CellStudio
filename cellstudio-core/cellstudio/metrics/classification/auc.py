from sklearn.metrics import roc_auc_score

from ..base import BaseMetric
from ..registry import MetricRegistry


@MetricRegistry.register("AUC")
class AUC(BaseMetric):
    def __init__(self, num_classes=2, **kwargs): self.num_classes = num_classes
    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:
        if y_prob is None: return 0.0
        if self.num_classes == 2:
            prob = y_prob if y_prob.ndim == 1 or y_prob.shape[1] == 1 else y_prob[:, 1]
            return float(roc_auc_score(y_true, prob))
        return float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
