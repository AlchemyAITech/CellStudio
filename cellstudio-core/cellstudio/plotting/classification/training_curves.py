import os

import matplotlib.pyplot as plt
import pandas as pd

from .. import init_plot_style
from ..base import BasePlotter
from ..registry import PlotterRegistry


@PlotterRegistry.register("Training_Curves")
class TrainingCurvesPlotter(BasePlotter):
    def __init__(self, **kwargs):
        pass # Inherited pattern, does not require targets
        
    def plot(self, save_dir: str, csv_path: str = None, **kwargs):
        if not csv_path or not os.path.exists(csv_path):
            print(f"[Warn] Missing training history log at {csv_path}")
            return
            
        df = pd.read_csv(csv_path)
        init_plot_style()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss Curve
        axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='royalblue')
        if 'val_loss' in df.columns:
            axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', color='darkorange')
        axes[0].set_title('Loss Curve Tracking'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend()
        
        # Acc Curve
        axes[1].plot(df['epoch'], df['train_acc'], label='Train Acc', color='seagreen')
        if 'val_acc' in df.columns:
            axes[1].plot(df['epoch'], df['val_acc'], label='Val Acc', color='crimson')
        axes[1].set_title('Validation Feedback Tracker'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].legend()
        
        # LR Curve
        if 'lr' in df.columns:
            axes[2].plot(df['epoch'], df['lr'], color='purple')
            axes[2].set_title('Learning Rate Decay Profile'); axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Learning Rate')
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curves.png"))
        plt.close()
