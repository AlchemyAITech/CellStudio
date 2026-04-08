import yaml
import glob
import os  # noqa: F401

configs = glob.glob('e:/workspace/AlchemyTech/CellStudio/configs/segmentation/*.yaml')

train_ds = {
    'type': 'CellposeSegmentationDataset',
    'data_root': 'datasets/segmentation/cellpose',
    'ann_file': 'splits/train_fold0.json',
    'pipeline': [
        {'type': 'LoadImageFromFile'},
        {'type': 'Resize', 'size': [512, 512]},
        {'type': 'RandomFlip', 'prob': 0.5},
        {'type': 'PackCellStudioInputs'}
    ]
}

val_ds = {
    'type': 'CellposeSegmentationDataset',
    'data_root': 'datasets/segmentation/cellpose',
    'ann_file': 'splits/val_fold0.json',
    'pipeline': [
        {'type': 'LoadImageFromFile'},
        {'type': 'Resize', 'size': [512, 512]},
        {'type': 'PackCellStudioInputs'}
    ]
}

for cfg_path in configs:
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    cfg['train_dataloader']['dataset'] = train_ds.copy()
    cfg['val_dataloader']['dataset'] = val_ds.copy()
        
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
        
    print(f"Fixed dataset blocks in {cfg_path}")
