import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from .base import BasePlotter
from .registry import PlotterRegistry

@PlotterRegistry.register('ROCPlotter')
class ROCPlotter(BasePlotter):
    def __init__(self, **kwargs):
        pass
        
    def plot(self, save_dir: str, y_true=None, y_pred=None, y_prob=None, **kwargs):
        if y_prob is None or y_true is None:
            print("[ROCPlotter] Skipping ROC plot as y_prob is unavailable.")
            return
            
        if isinstance(y_true, torch.Tensor): y_true = y_true.numpy()
        if isinstance(y_prob, torch.Tensor): y_prob = y_prob.numpy()
        
        # Binary classification handling
        if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        elif len(y_prob.shape) > 1:
            print("[ROCPlotter] Cannot plot binary ROC for multi-class prob tensor.")
            return
            
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "roc_curve.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[ROCPlotter] Saved ROC Curve to {save_path}")

@PlotterRegistry.register('ConfusionMatrixPlotter')
class ConfusionMatrixPlotter(BasePlotter):
    def __init__(self, **kwargs):
        pass
        
    def plot(self, save_dir: str, y_true=None, y_pred=None, y_prob=None, **kwargs):
        if y_true is None or y_pred is None:
            print("[ConfusionMatrixPlotter] Skipping due to missing predictions.")
            return
        if isinstance(y_true, torch.Tensor): y_true = y_true.numpy()
        if isinstance(y_pred, torch.Tensor): y_pred = y_pred.numpy()
        
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        
        fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title('Confusion Matrix')
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[ConfusionMatrixPlotter] Saved Confusion Matrix to {save_path}")
