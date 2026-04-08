import datetime
import json
import logging
import os
import time

from .base import Hook
from .registry import HOOK_REGISTRY


@HOOK_REGISTRY.register('TextLoggerHook')
class TextLoggerHook(Hook):
    def __init__(self, interval=1):
        self.interval = interval
        self.logger = logging.getLogger('cellstudio')
        self.logger.setLevel(logging.INFO)
        self.configured = False
        self.json_log_path = None
        self.iter_start_time = None
        
    def _setup_logging(self, runner):
        if not self.configured:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Always add console handler if not present
            has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in self.logger.handlers)
            if not has_console:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
            
            # Always add file handler for work_dir
            if hasattr(runner, 'work_dir') and runner.work_dir:
                os.makedirs(runner.work_dir, exist_ok=True)
                log_path = os.path.join(runner.work_dir, 'training.log')
                # Check if file handler for this path already exists
                has_file = any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path) for h in self.logger.handlers)
                if not has_file:
                    file_handler = logging.FileHandler(log_path)
                    file_handler.setLevel(logging.INFO)
                    file_handler.setFormatter(formatter)
                    self.logger.addHandler(file_handler)
                self.json_log_path = os.path.join(runner.work_dir, 'scalars.json')
            self.configured = True

    def before_run(self, runner, **kwargs):
        self._setup_logging(runner)

    def _dump_json(self, record):
        if self.json_log_path:
            with open(self.json_log_path, 'a') as f:
                f.write(json.dumps(record) + '\n')

    def before_train_iter(self, runner, **kwargs):
        self.iter_start_time = time.time()

    def after_train_iter(self, runner, **kwargs):
        batch_time = time.time() - self.iter_start_time
        if runner.iter % self.interval == 0:
            lr = runner.optimizer.param_groups[0]['lr'] if hasattr(runner, 'optimizer') else 0.0
            
            loss_dict = {}
            for k, v in runner.outputs.items():
                if 'loss' in k or 'acc' in k:
                    if hasattr(v, 'item'):
                        try:
                            # if it's a multi-element tensor, taking mean or sum is usually better, but for logging we can convert to list or take mean
                            loss_dict[k] = v.item() if v.numel() == 1 else v.mean().item()
                        except Exception:
                            loss_dict[k] = str(v)
                    else:
                        loss_dict[k] = v
            loss_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in loss_dict.items())
            
            iters_this_epoch = len(runner.train_dataloader)
            remain_iters = iters_this_epoch - runner.inner_iter - 1
            eta_str = str(datetime.timedelta(seconds=int(remain_iters * batch_time)))
            
            self.logger.info(f"Epoch [{runner.epoch}/{runner.max_epochs}] "
                             f"Iter [{runner.inner_iter}/{iters_this_epoch}] - "
                             f"eta: {eta_str} - time: {batch_time:.3f}s - lr: {lr:.2e} - {loss_str}")
            
            record = {'mode': 'train', 'epoch': runner.epoch, 'iter': runner.iter, 'lr': lr, 'time': batch_time, **loss_dict}
            self._dump_json(record)

    def after_val_epoch(self, runner, **kwargs):
        if hasattr(runner, 'val_metrics'):
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in runner.val_metrics.items())
            self.logger.info(f"Val Epoch [{runner.epoch}/{runner.max_epochs}] - {metrics_str}")
            record = {'mode': 'val', 'epoch': runner.epoch, **runner.val_metrics}
            self._dump_json(record)
