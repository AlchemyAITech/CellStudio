
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve

from .. import init_plot_style
from ..base import BasePlotter
from ..registry import PlotterRegistry


@PlotterRegistry.register("PR_Curve")
class PRCurvePlotter(BasePlotter):
    def __init__(self, y_true, y_pred, y_prob=None, **kwargs):
        self.y_true = np.asarray(y_true)
        self.pos_prob = y_prob if y_prob is None or np.asarray(y_prob).ndim == 1 or np.asarray(y_prob).shape[1] == 1 else np.asarray(y_prob)[:, 1]
    def plot(self, save_dir: str, **kwargs):
        if self.pos_prob is None: return
        init_plot_style()
        precision, recall, _ = precision_recall_curve(self.y_true, self.pos_prob)
        pr_auc = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(save_dir, "pr_curve.png")); plt.close()
