from abc import ABC, abstractmethod

class BasePlotter(ABC):
    @abstractmethod
    def __init__(self, **kwargs): pass
    @abstractmethod
    def plot(self, save_dir: str, y_true=None, y_pred=None, y_prob=None, **kwargs): pass
