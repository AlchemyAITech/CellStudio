
import os

import matplotlib.pyplot as plt
import numpy as np

from .. import init_plot_style
from ..base import BasePlotter
from ..registry import PlotterRegistry


@PlotterRegistry.register("DCA_Curve")
class DCAPlotter(BasePlotter):
    def __init__(self, y_true, y_pred, y_prob=None, **kwargs):
        self.y_true = np.asarray(y_true)
        self.pos_prob = y_prob if y_prob is None or np.asarray(y_prob).ndim == 1 or np.asarray(y_prob).shape[1] == 1 else np.asarray(y_prob)[:, 1]
    def plot(self, save_dir: str, **kwargs):
        if self.pos_prob is None: return
        init_plot_style()
        thresholds = np.linspace(0.01, 0.99, 50)
        net_benefits = []
        prevalence = np.mean(self.y_true)
        for t in thresholds:
            tp = np.sum((self.pos_prob >= t) & (self.y_true == 1))
            fp = np.sum((self.pos_prob >= t) & (self.y_true == 0))
            nb = (tp / len(self.y_true)) - (fp / len(self.y_true)) * (t / (1 - t))
            net_benefits.append(nb)
        plt.figure()
        plt.plot(thresholds, net_benefits, label='Model', color='red', lw=2)
        plt.plot(thresholds, [prevalence]*len(thresholds), label='Treat All', color='grey', ls='--')
        plt.plot(thresholds, [0]*len(thresholds), label='Treat None', color='black', ls=':')
        plt.ylim(-0.05, max(prevalence+0.1, max(net_benefits)+0.1))
        plt.xlabel('Threshold Probability'); plt.ylabel('Net Benefit'); plt.title('Decision Curve Analysis (DCA)')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(save_dir, "dca_curve.png")); plt.close()
