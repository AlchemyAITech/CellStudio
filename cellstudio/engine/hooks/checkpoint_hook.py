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
        if runner.epoch % self.interval == 0:
            self._save(runner, f"epoch_{runner.epoch}.pth")

    def after_val_epoch(self, runner, **kwargs):
        if hasattr(runner, 'val_metrics') and self.save_best in runner.val_metrics:
            score = runner.val_metrics[self.save_best]
            is_best = score > self.best_score if self.rule == 'greater' else score < self.best_score
            if is_best:
                self.best_score = score
                self._save(runner, "best.pth")

    def _save(self, runner, filename):
        os.makedirs(runner.work_dir, exist_ok=True)
        path = os.path.join(runner.work_dir, filename)
        torch.save(runner.model.state_dict(), path)
