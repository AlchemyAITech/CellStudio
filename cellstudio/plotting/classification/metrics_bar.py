
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..registry import PlotterRegistry
from ..base import BasePlotter
from .. import init_plot_style

@PlotterRegistry.register("Metrics_Bar")
class MetricsBarPlotter(BasePlotter):
    def __init__(self, y_true, y_pred, y_prob=None, **kwargs): pass
    def plot(self, save_dir: str, metrics_dict=None, **kwargs):
        if not metrics_dict: return
        init_plot_style()
        plt.figure(figsize=(10, 5))
        keys, vals = list(metrics_dict.keys()), list(metrics_dict.values())
        sns.barplot(x=keys, y=vals, hue=keys, legend=False, palette="viridis")
        plt.ylim(0, 1.1)
        for i, v in enumerate(vals): plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
        plt.title('Classification Metrics Summary')
        plt.savefig(os.path.join(save_dir, "metrics_bar.png")); plt.close()
