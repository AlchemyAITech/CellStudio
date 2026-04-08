import os

import matplotlib.pyplot as plt

from .base import Hook
from .registry import HOOK_REGISTRY


@HOOK_REGISTRY.register('TrainingProgressPlotterHook')
class TrainingProgressPlotterHook(Hook):
    """
    Plots training loss and validation accuracy curves incrementally across epochs.
    """
    def __init__(self, out_dir: str = 'work_dirs/plots', **kwargs):
        self.out_dir = out_dir
        self.epochs = []
        self.train_losses = []
        self.val_accuracies = []
        self._current_epoch_losses = []

    def before_run(self, runner, **kwargs):
        # Rebind to runner's actual work_dir if available
        if hasattr(runner, 'work_dir') and runner.work_dir:
            self.out_dir = runner.work_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def after_train_iter(self, runner, **kwargs):
        if 'loss' in runner.outputs:
            self._current_epoch_losses.append(runner.outputs['loss'].item())

    def after_train_epoch(self, runner, **kwargs):
        if self._current_epoch_losses:
            avg_loss = sum(self._current_epoch_losses) / len(self._current_epoch_losses)
            self.train_losses.append(avg_loss)
            self.epochs.append(runner.epoch + 1)
            self._current_epoch_losses = []
            self._plot_curves()

    def after_val_epoch(self, runner, **kwargs):
        if hasattr(runner, 'val_metrics'):
            # Fetch primary metrics
            acc = runner.val_metrics.get('accuracy', 0.0)
            self.val_accuracies.append(acc)
            self._plot_curves()

    def _plot_curves(self):
        fig, ax1 = plt.subplots(figsize=(8, 6))

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss', color=color)
        ax1.plot(self.epochs, self.train_losses, color=color, marker='o', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        if self.val_accuracies:
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Validation Accuracy', color=color)
            
            # Align lists if validation happens less frequently
            val_epochs = self.epochs[:len(self.val_accuracies)]
            ax2.plot(val_epochs, self.val_accuracies, color=color, marker='s', label='Val Accuracy')
            ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title("Training Loss & Validation Accuracy")
        plt.savefig(os.path.join(self.out_dir, 'training_progressCurve.png'), dpi=300)
        plt.close(fig)
