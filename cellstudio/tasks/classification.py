from .base import BaseTask
from .registry import TASK_REGISTRY
from ..datasets.registry import DatasetRegistry

# Automatically import implementations so globals register the components
from ..datasets.collate import pseudo_collate
from torch.utils.data import DataLoader

@TASK_REGISTRY.register('ClassificationTask')
class ClassificationTask(BaseTask):
    """
    Spawns Classification tasks.
    """
    def build_datasets(self):
        train_cfg = self.cfg.get('train_dataloader')
        val_cfg = self.cfg.get('val_dataloader')
        
        if train_cfg:
            ds_cfg = train_cfg.get('dataset')
            ds_type = ds_cfg.pop('type', 'StandardClassificationDataset')
            ds_cls = DatasetRegistry.get(ds_type)
            train_dataset = ds_cls(**ds_cfg)
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=train_cfg.get('batch_size', 2),
                num_workers=train_cfg.get('num_workers', 2),
                shuffle=True,
                collate_fn=pseudo_collate,
                drop_last=True
            )
            
        if val_cfg:
            ds_cfg = val_cfg.get('dataset')
            ds_type = ds_cfg.pop('type', 'StandardClassificationDataset')
            ds_cls = DatasetRegistry.get(ds_type)
            val_dataset = ds_cls(**ds_cfg)
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=val_cfg.get('batch_size', 1),
                num_workers=val_cfg.get('num_workers', 2),
                shuffle=False,
                collate_fn=pseudo_collate
            )
