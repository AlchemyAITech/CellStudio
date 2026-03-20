
from typing import Dict, Type
from .base import BasePlotter

class PlotterRegistry:
    _registry: Dict[str, Type[BasePlotter]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(plotter_cls: Type[BasePlotter]):
            cls._registry[name] = plotter_cls
            return plotter_cls
        return wrapper

    @classmethod
    def get(cls, name: str) -> Type[BasePlotter]:
        if name not in cls._registry: raise KeyError(f"Plotter '{name}' not found.")
        return cls._registry[name]

    @classmethod
    def build(cls, cfg: dict):
        if cfg is None: return None
        cfg_copy = cfg.copy()
        plugin_type = cfg_copy.pop('type')
        return cls.get(plugin_type)(**cfg_copy)

class PlotterCollection:
    def __init__(self, plotter_names: list, y_true, y_pred, y_prob=None, **kwargs):
        self.plotters = {n: PlotterRegistry.get(n)(y_true, y_pred, y_prob, **kwargs) for n in plotter_names}
        
    def generate_all(self, save_dir: str, **kwargs):
        import os
        os.makedirs(save_dir, exist_ok=True)
        for name, plotter in self.plotters.items():
            plotter.plot(save_dir, **kwargs)
