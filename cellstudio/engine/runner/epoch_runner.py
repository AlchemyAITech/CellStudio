from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader

from .base_runner import BaseRunner

class EpochBasedRunner(BaseRunner):
    """
    The Crown Jewel of the Zenith Architecture: EpochBasedRunner.
    Completely isolated from model specific logic; orchestrated entirely via hooks.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 train_dataloader: Optional[DataLoader] = None,
                 val_dataloader: Optional[DataLoader] = None,
                 max_epochs: int = 100,
                 work_dir: str = None,
                 meta: Dict = None,
                 use_amp: bool = False,
                 cfg: Dict = None):
        super().__init__(model, optimizer, work_dir, meta, use_amp)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_epochs = max_epochs
        self.cfg = cfg or {}
        
        if self.train_dataloader is not None:
            self.max_iters = self.max_epochs * len(self.train_dataloader)

    def train(self):
        self.model.train()
        
        self.call_hook('before_run')
        
        # Setup base learning rates for schedulers
        if self.optimizer:
            base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        else:
            base_lrs = []
            
        sched_cfg = self.cfg.get('param_scheduler', [])
        
        while self.epoch < self.max_epochs:
            self.call_hook('before_train_epoch')
            
            for i, data_batch in enumerate(self.train_dataloader):
                self.inner_iter = i
                
                # Manual LR Scheduler calculation
                if self.optimizer and sched_cfg:
                    lr_factor = 1.0
                    for cfg_s in sched_cfg:
                        if cfg_s.get('type') == 'LinearLR':
                            begin = cfg_s.get('begin', 0)
                            end = cfg_s.get('end', 500)
                            start_factor = cfg_s.get('start_factor', 0.001)
                            if self.iter <= end:
                                progress = (self.iter - begin) / max(1, end - begin)
                                progress = min(max(progress, 0.0), 1.0)
                                factor = start_factor + (1.0 - start_factor) * progress
                                lr_factor *= factor
                        elif cfg_s.get('type') == 'MultiStepLR':
                            milestones = cfg_s.get('milestones', [40, 80])
                            gamma = cfg_s.get('gamma', 0.1)
                            for m in milestones:
                                if self.epoch >= m:
                                    lr_factor *= gamma
                    
                    for idx, group in enumerate(self.optimizer.param_groups):
                        group['lr'] = base_lrs[idx] * lr_factor
                
                self.call_hook('before_train_iter', batch_idx=i, data_batch=data_batch)
                
                # Model forward with optional AMP
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    # Adapter logic abstracts whether this is YOLO, Cellpose, or ResNet
                    outputs = self.model.forward_train(data_batch['imgs'], data_batch.get('data_samples'))

                # Ensure outputs dictionary contains 'loss' metric
                self.outputs = outputs
                self.call_hook('after_train_iter', batch_idx=i, data_batch=data_batch, outputs=outputs)

                self.iter += 1

            self.call_hook('after_train_epoch')

            # Validation runs every val_interval epochs (and always on last epoch)
            val_interval = self.cfg.get('runner', {}).get('val_interval', 1)
            is_last = (self.epoch + 1 >= self.max_epochs)
            if self.val_dataloader and ((self.epoch + 1) % val_interval == 0 or is_last):
                self.val()

            self.epoch += 1

        self.call_hook('after_run')

    @torch.no_grad()
    def val(self):
        self.call_hook('before_val_epoch')
        val_losses = []

        for i, data_batch in enumerate(self.val_dataloader):
            self.data_batch = data_batch
            self.call_hook('before_val_iter', batch_idx=i, data_batch=data_batch)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                # Compute val_loss: YOLO needs train mode to produce raw features
                # for the loss function (eval mode runs NMS which breaks loss).
                # No gradients are computed since we're under @torch.no_grad().
                self.model.train()
                try:
                    loss_dict = self.model.forward_train(data_batch['imgs'], data_batch.get('data_samples'))
                    if 'loss' in loss_dict:
                        val_losses.append(loss_dict['loss'].item())
                except Exception:
                    pass  # Silently skip if loss computation fails

                # Compute predictions for mAP evaluation
                self.model.eval()
                outputs = self.model.forward_test(data_batch['imgs'], data_batch.get('data_samples'))
            
            self.outputs = outputs
            self.call_hook('after_val_iter', batch_idx=i, data_batch=data_batch, outputs=outputs)
            
        metrics = {}
        if val_losses:
            metrics['val_loss'] = sum(val_losses) / len(val_losses)
            
        # Evaluator hook populates other metrics into self.val_metrics
        self.call_hook('after_val_epoch', metrics=metrics) 
        self.model.train()
