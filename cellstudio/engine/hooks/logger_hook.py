import logging
import os
import json
from .base import Hook
from .registry import HOOK_REGISTRY

@HOOK_REGISTRY.register('TextLoggerHook')
class TextLoggerHook(Hook):
    def __init__(self, interval=10):
        self.interval = interval
        self.logger = logging.getLogger('cellstudio')
        self.logger.setLevel(logging.INFO)
        self.configured = False
        self.json_log_path = None
        
    def _setup_logging(self, runner):
        if not self.configured:
            # Add Console and File Handlers if missing
            if not self.logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
                
                if hasattr(runner, 'work_dir') and runner.work_dir:
                    os.makedirs(runner.work_dir, exist_ok=True)
                    file_handler = logging.FileHandler(os.path.join(runner.work_dir, 'training.log'))
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

    def after_train_iter(self, runner, **kwargs):
        if runner.iter % self.interval == 0:
            loss_dict = {k: v.item() if hasattr(v, 'item') else v for k, v in runner.outputs.items() if 'loss' in k}
            loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items())
            self.logger.info(f"Epoch [{runner.epoch}/{runner.max_epochs}] Iter [{runner.inner_iter}/{len(runner.train_dataloader)}] - {loss_str}")
            
            # Serialize structured JSON
            record = {'mode': 'train', 'epoch': runner.epoch, 'iter': runner.iter, **loss_dict}
            self._dump_json(record)

    def after_val_epoch(self, runner, **kwargs):
        if hasattr(runner, 'val_metrics'):
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in runner.val_metrics.items())
            self.logger.info(f"Val Epoch [{runner.epoch}/{runner.max_epochs}] - {metrics_str}")
            
            record = {'mode': 'val', 'epoch': runner.epoch, **runner.val_metrics}
            self._dump_json(record)
