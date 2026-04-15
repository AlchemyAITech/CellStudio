
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .. import init_plot_style
from ..base import BasePlotter
from ..registry import PlotterRegistry


@PlotterRegistry.register("Confusion_Matrix")
class ConfusionMatrixPlotter(BasePlotter):
    def __init__(self, y_true, y_pred, y_prob=None, **kwargs):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
    def plot(self, save_dir: str, **kwargs):
        init_plot_style()
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Raw Counts)'); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_raw.png')); plt.close()
        
        cm_norm = confusion_matrix(self.y_true, self.y_pred, normalize='true')
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Confusion Matrix (Normalized)'); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_norm.png')); plt.close()
