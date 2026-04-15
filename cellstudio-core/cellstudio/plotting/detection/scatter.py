import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cellstudio.metrics.detection.core import DetMatchCache

from ..base import BasePlotter
from ..registry import PlotterRegistry


@PlotterRegistry.register("det_medical_scatter")
class DetMedicalScatterPlotter(BasePlotter):
    def __init__(self, y_true, y_pred, y_prob=None, **kwargs):
        state = DetMatchCache.get(y_true, y_pred, y_prob, iou_thresh=kwargs.get('iou_thresh', 0.5))
        self.t_counts, self.p_counts = np.array(state["true_counts"]), np.array(state["pred_counts"])
    def plot(self, save_dir: str, **kwargs):
        if len(self.t_counts) == 0: return
        plt.figure(figsize=(8, 6)); sns.regplot(x=self.t_counts, y=self.p_counts)
        plt.savefig(os.path.join(save_dir, "count_scatter.png")); plt.close()
