import yaml
import glob
import os  # noqa: F401

configs = glob.glob('e:/workspace/AlchemyTech/CellStudio/configs/segmentation/*.yaml')

for cfg_path in configs:
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Inject Resize into pipelines
    new_train_pl = []
    for t in cfg['dataset']['pipeline']:
        new_train_pl.append(t)
        if t['type'] == 'LoadImageFromFile':
            new_train_pl.append({'type': 'Resize', 'size': [512, 512]})
    cfg['dataset']['pipeline'] = new_train_pl
    
    new_val_pl = []
    for t in cfg['val_dataset']['pipeline']:
        new_val_pl.append(t)
        if t['type'] == 'LoadImageFromFile':
            new_val_pl.append({'type': 'Resize', 'size': [512, 512]})
    cfg['val_dataset']['pipeline'] = new_val_pl
    
    # Enforce 30 epochs
    cfg['runner']['max_epochs'] = 30
    if 'scheduler' in cfg['runner'] and 'T_max' in cfg['runner']['scheduler']:
        cfg['runner']['scheduler']['T_max'] = 30
        
    # Enforce batch sizes
    cfg['dataloader']['batch_size'] = 4
    cfg['dataloader']['num_workers'] = 4
    
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
        
    print(f"Updated {cfg_path}")
