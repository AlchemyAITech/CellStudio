from abc import ABC, abstractmethod

class BaseMetric(ABC):
    @abstractmethod
    def __init__(self, **kwargs): pass
    @abstractmethod
    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float: pass
