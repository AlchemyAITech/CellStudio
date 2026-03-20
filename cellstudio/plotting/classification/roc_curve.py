
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from ..registry import PlotterRegistry
from ..base import BasePlotter
from .. import init_plot_style

@PlotterRegistry.register("ROC_Curve")
class ROCCurvePlotter(BasePlotter):
    def __init__(self, y_true, y_pred, y_prob=None, **kwargs):
        self.y_true = np.asarray(y_true)
        self.pos_prob = y_prob if y_prob is None or np.asarray(y_prob).ndim == 1 or np.asarray(y_prob).shape[1] == 1 else np.asarray(y_prob)[:, 1]
    def plot(self, save_dir: str, **kwargs):
        if self.pos_prob is None: return
        init_plot_style()
        fpr, tpr, _ = roc_curve(self.y_true, self.pos_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, "roc_curve.png")); plt.close()
