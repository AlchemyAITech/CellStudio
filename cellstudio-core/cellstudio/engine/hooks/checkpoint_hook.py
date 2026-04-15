import os

import torch

from .base import Hook
from .registry import HOOK_REGISTRY


@HOOK_REGISTRY.register('CheckpointHook')
class CheckpointHook(Hook):
    def __init__(self, interval=10, save_best='mAP50', rule='greater'):
        self.interval = interval
        self.save_best = save_best
        self.rule = rule
        self.best_score = float('-inf') if rule == 'greater' else float('inf')

    def after_train_epoch(self, runner, **kwargs):
        # Always overwrite latest.pth
        self._save(runner, "latest.pth")

    def after_val_epoch(self, runner, **kwargs):
        if hasattr(runner, 'val_metrics') and self.save_best in runner.val_metrics:
            score = runner.val_metrics[self.save_best]
            is_best = score > self.best_score if self.rule == 'greater' else score < self.best_score
            if is_best:
                self.best_score = score
                runner.is_best_epoch = True
                self._save(runner, "best.pth")
                
                # Also save predictions for the best epoch
                import shutil
                pred_file = os.path.join(runner.work_dir, 'predictions.pkl')
                best_pred_file = os.path.join(runner.work_dir, 'best_predictions.pkl')
                if os.path.exists(pred_file):
                    shutil.copy(pred_file, best_pred_file)
            else:
                runner.is_best_epoch = False

    def _save(self, runner, filename):
        os.makedirs(runner.work_dir, exist_ok=True)
        path = os.path.join(runner.work_dir, filename)
        torch.save(runner.model.state_dict(), path)
