import os
import shutil

base_dir = r"e:\workspace\AlchemyTech\CellStudio\cellstudio\metrics"
cls_dir = os.path.join(base_dir, "classification")
os.makedirs(cls_dir, exist_ok=True)

# Cleanup
for f in ["base_metrics.py", "classification_metrics.py"]:
    p = os.path.join(base_dir, f)
    if os.path.exists(p): os.remove(p)

def write_f(p, c):
    with open(p, "w", encoding="utf-8") as f: f.write(c)

write_f(os.path.join(base_dir, "base.py"), '''from abc import ABC, abstractmethod\n\nclass BaseMetric(ABC):\n    @abstractmethod\n    def __init__(self, **kwargs): pass\n    @abstractmethod\n    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float: pass\n''')

write_f(os.path.join(base_dir, "registry.py"), '''from typing import Dict, Type, Any\nfrom .base import BaseMetric\n\nclass MetricRegistry:\n    _registry: Dict[str, Type[BaseMetric]] = {}\n    @classmethod\n    def register(cls, name: str):\n        def wrapper(metric_cls: Type[BaseMetric]):\n            cls._registry[name] = metric_cls\n            return metric_cls\n        return wrapper\n    @classmethod\n    def get(cls, name: str) -> Type[BaseMetric]:\n        if name not in cls._registry: raise KeyError(f"Metric '{name}' not found.")\n        return cls._registry[name]\n\nclass MetricCollection:\n    def __init__(self, metric_names: list, **kwargs):\n        self.metrics = {n: MetricRegistry.get(n)(**kwargs) for n in metric_names}\n    def compute_all(self, y_true, y_pred, y_prob=None, **kwargs) -> Dict[str, float]:\n        results = {}\n        for n, m in self.metrics.items():\n            try: results[n] = m.compute(y_true, y_pred, y_prob, **kwargs)\n            except Exception: results[n] = 0.0\n        return results\n''')

write_f(os.path.join(base_dir, "__init__.py"), '''from .base import BaseMetric\nfrom .registry import MetricRegistry, MetricCollection\nfrom . import classification\n''')

write_f(os.path.join(cls_dir, "accuracy.py"), '''from sklearn.metrics import accuracy_score\nfrom ..registry import MetricRegistry\nfrom ..base import BaseMetric\n@MetricRegistry.register("Accuracy")\nclass Accuracy(BaseMetric):\n    def __init__(self, **kwargs): pass\n    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:\n        return float(accuracy_score(y_true, y_pred))\n''')
write_f(os.path.join(cls_dir, "precision.py"), '''from sklearn.metrics import precision_score\nfrom ..registry import MetricRegistry\nfrom ..base import BaseMetric\n@MetricRegistry.register("Precision")\nclass Precision(BaseMetric):\n    def __init__(self, num_classes=2, **kwargs): self.avg = 'binary' if num_classes == 2 else 'macro'\n    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:\n        return float(precision_score(y_true, y_pred, average=self.avg, zero_division=0))\n''')
write_f(os.path.join(cls_dir, "recall.py"), '''from sklearn.metrics import recall_score\nfrom ..registry import MetricRegistry\nfrom ..base import BaseMetric\n@MetricRegistry.register("Recall")\nclass Recall(BaseMetric):\n    def __init__(self, num_classes=2, **kwargs): self.avg = 'binary' if num_classes == 2 else 'macro'\n    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:\n        return float(recall_score(y_true, y_pred, average=self.avg, zero_division=0))\n''')
write_f(os.path.join(cls_dir, "f1_score.py"), '''from sklearn.metrics import f1_score\nfrom ..registry import MetricRegistry\nfrom ..base import BaseMetric\n@MetricRegistry.register("F1_Score")\nclass F1Score(BaseMetric):\n    def __init__(self, num_classes=2, **kwargs): self.avg = 'binary' if num_classes == 2 else 'macro'\n    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:\n        return float(f1_score(y_true, y_pred, average=self.avg, zero_division=0))\n''')
write_f(os.path.join(cls_dir, "auc.py"), '''from sklearn.metrics import roc_auc_score\nfrom ..registry import MetricRegistry\nfrom ..base import BaseMetric\n@MetricRegistry.register("AUC")\nclass AUC(BaseMetric):\n    def __init__(self, num_classes=2, **kwargs): self.num_classes = num_classes\n    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:\n        if y_prob is None: return 0.0\n        if self.num_classes == 2:\n            prob = y_prob if y_prob.ndim == 1 or y_prob.shape[1] == 1 else y_prob[:, 1]\n            return float(roc_auc_score(y_true, prob))\n        return float(roc_auc_score(y_true, y_prob, multi_class='ovr'))\n''')
write_f(os.path.join(cls_dir, "pr_auc.py"), '''from sklearn.metrics import average_precision_score\nfrom ..registry import MetricRegistry\nfrom ..base import BaseMetric\n@MetricRegistry.register("PR_AUC")\nclass PRAUC(BaseMetric):\n    def __init__(self, num_classes=2, **kwargs): self.num_classes = num_classes\n    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:\n        if y_prob is None or self.num_classes != 2: return 0.0\n        prob = y_prob if y_prob.ndim == 1 or y_prob.shape[1] == 1 else y_prob[:, 1]\n        return float(average_precision_score(y_true, prob))\n''')
write_f(os.path.join(cls_dir, "kappa.py"), '''from sklearn.metrics import cohen_kappa_score\nfrom ..registry import MetricRegistry\nfrom ..base import BaseMetric\n@MetricRegistry.register("Kappa")\nclass Kappa(BaseMetric):\n    def __init__(self, **kwargs): pass\n    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:\n        return float(cohen_kappa_score(y_true, y_pred))\n''')
write_f(os.path.join(cls_dir, "icc.py"), '''import numpy as np\nimport pandas as pd\nfrom ..registry import MetricRegistry\nfrom ..base import BaseMetric\n@MetricRegistry.register("ICC")\nclass ICC(BaseMetric):\n    def __init__(self, **kwargs): pass\n    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:\n        try:\n            import pingouin as pg\n            n = len(y_true)\n            df = pd.DataFrame({'Target': np.concatenate([np.arange(n), np.arange(n)]), 'Rater': np.concatenate([np.zeros(n), np.ones(n)]), 'Score': np.concatenate([y_true, y_pred])})\n            icc_res = pg.intraclass_corr(data=df, targets='Target', raters='Rater', ratings='Score')\n            return float(icc_res.set_index('Type').loc['ICC2', 'ICC'])\n        except Exception:\n            return 0.0\n''')

write_f(os.path.join(cls_dir, "__init__.py"), '''from .accuracy import Accuracy\nfrom .precision import Precision\nfrom .recall import Recall\nfrom .f1_score import F1Score\nfrom .auc import AUC\nfrom .pr_auc import PRAUC\nfrom .kappa import Kappa\nfrom .icc import ICC\n''')

print("Refactoring Complete!")
