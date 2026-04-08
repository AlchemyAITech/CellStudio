from torch.utils.data import DataLoader

from ..datasets.collate import pseudo_collate
from .base import BaseTask
from .registry import TASK_REGISTRY


@TASK_REGISTRY.register('InstanceSegmentationTask')
class InstanceSegmentationTask(BaseTask):
    """
    Spawns Instance Segmentation models using the Zenith Runner components.
    Supports CellposeSegmentationDataset.
    """
    def _build_dataset(self, ds_cfg_raw):
        ds_cfg = ds_cfg_raw.copy()
        ds_type = ds_cfg.pop('type', 'CellposeSegmentationDataset')
        
        if ds_type == 'CellposeSegmentationDataset':
            from ..datasets.segmentation import CellposeSegmentationDataset
            return CellposeSegmentationDataset(**ds_cfg)
        else:
            raise ValueError(f"Unknown segmentation dataset type: {ds_type}")
    
    def build_datasets(self):
        train_cfg = self.cfg.get('train_dataloader')
        val_cfg = self.cfg.get('val_dataloader')
        
        if train_cfg:
            train_dataset = self._build_dataset(train_cfg.get('dataset'))
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=train_cfg.get('batch_size', 2),
                num_workers=train_cfg.get('num_workers', 2),
                shuffle=True,
                collate_fn=pseudo_collate,
                drop_last=True
            )
            
        if val_cfg:
            val_dataset = self._build_dataset(val_cfg.get('dataset'))
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=val_cfg.get('batch_size', 1),
                num_workers=val_cfg.get('num_workers', 2),
                shuffle=False,
                collate_fn=pseudo_collate
            )
