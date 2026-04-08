import yaml
import glob
import os  # noqa: F401

configs = glob.glob('e:/workspace/AlchemyTech/CellStudio/configs/segmentation/*.yaml')

for cfg_path in configs:
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    new_cfg = {}
    
    # 1. Base fields
    if '_base_' in cfg:
        new_cfg['_base_'] = cfg['_base_']
        
    # 2. Task
    new_cfg['task'] = {'type': 'InstanceSegmentationTask'}
    
    # 3. Model
    new_cfg['model'] = cfg.get('model', {})
    
    # 4. Train Dataloader
    new_cfg['train_dataloader'] = {
        'batch_size': cfg.get('dataloader', {}).get('batch_size', 4),
        'num_workers': 0,
        'dataset': cfg.get('dataset', {})
    }
    
    # 5. Val Dataloader
    new_cfg['val_dataloader'] = {
        'batch_size': cfg.get('dataloader', {}).get('batch_size', 4),
        'num_workers': 0,
        'dataset': cfg.get('val_dataset', {})
    }
    
    # 6. Evaluator
    new_cfg['val_evaluator'] = cfg.get('val_evaluator', {})
    
    # 7. Runner & Optimizer
    runner = cfg.get('runner', {})
    
    # Extract optimizer, scheduler from runner if they are inside
    if 'optimizer' in runner:
        new_cfg['optim_wrapper'] = {'optimizer': runner.pop('optimizer')}
    else:
        new_cfg['optim_wrapper'] = {'optimizer': cfg.get('optim_wrapper', {}).get('optimizer', {})}
        
    if 'scheduler' in runner:
        new_cfg['param_scheduler'] = [runner.pop('scheduler')]
    elif 'param_scheduler' in cfg:
        new_cfg['param_scheduler'] = cfg['param_scheduler']
        
    # Pop warmup
    if 'warmup' in runner:
        runner.pop('warmup')
        
    new_cfg['runner'] = runner
    
    if 'log_config' in cfg:
        new_cfg['log_config'] = cfg['log_config']
        
    if 'env' in cfg:
        new_cfg['env'] = cfg['env']
    else:
        new_cfg['env'] = {'cudnn_benchmark': True, 'device': 'cuda'}

    with open(cfg_path, 'w') as f:
        yaml.dump(new_cfg, f, sort_keys=False)
        
    print(f"Migrated {cfg_path} to Zenith Omega Architecture Standards.")
