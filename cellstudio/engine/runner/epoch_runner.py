from typing import Any, Dict, Optional
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
                 use_amp: bool = False):
        super().__init__(model, optimizer, work_dir, meta, use_amp)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_epochs = max_epochs
        
        if self.train_dataloader is not None:
            self.max_iters = self.max_epochs * len(self.train_dataloader)

    def train(self):
        self.model.train()
        self.call_hook('before_run')
        
        while self.epoch < self.max_epochs:
            self.call_hook('before_train_epoch')
            
            for i, data_batch in enumerate(self.train_dataloader):
                self.inner_iter = i
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

            # Validation triggered implicitly at epoch end
            if self.val_dataloader:
                self.val()

            self.epoch += 1

        self.call_hook('after_run')

    @torch.no_grad()
    def val(self):
        self.model.eval()
        self.call_hook('before_val_epoch')

        for i, data_batch in enumerate(self.val_dataloader):
            self.data_batch = data_batch
            self.call_hook('before_val_iter', batch_idx=i, data_batch=data_batch)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model.forward_test(data_batch['imgs'], data_batch.get('data_samples'))
            
            self.outputs = outputs
            self.call_hook('after_val_iter', batch_idx=i, data_batch=data_batch, outputs=outputs)
            
        # Evaluator hook populates metrics
        self.call_hook('after_val_epoch', metrics={}) 
        self.model.train()
