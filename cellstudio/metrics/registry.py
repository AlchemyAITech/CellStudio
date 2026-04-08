from typing import Dict, Type
from .base import BaseMetric

class MetricRegistry:
    _registry: Dict[str, Type[BaseMetric]] = {}
    @classmethod
    def register(cls, name: str):
        def wrapper(metric_cls: Type[BaseMetric]):
            cls._registry[name] = metric_cls
            return metric_cls
        return wrapper
    @classmethod
    def get(cls, name: str) -> Type[BaseMetric]:
        if name not in cls._registry: raise KeyError(f"Metric '{name}' not found.")
        return cls._registry[name]

    @classmethod
    def build(cls, cfg: dict):
        if cfg is None: return None
        cfg_copy = cfg.copy()
        plugin_type = cfg_copy.pop('type')
        return cls.get(plugin_type)(**cfg_copy)

class MetricCollection:
    def __init__(self, metric_names: list, **kwargs):
        self.metrics = {n: MetricRegistry.get(n)(**kwargs) for n in metric_names}
    def compute_all(self, y_true, y_pred, y_prob=None, **kwargs) -> Dict[str, float]:
        results = {}
        for n, m in self.metrics.items():
            try: results[n] = m.compute(y_true, y_pred, y_prob, **kwargs)
            except Exception: results[n] = 0.0
        return results
