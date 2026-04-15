import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cellstudio.engine.config.config import Config
from cellstudio.tasks.registry import TASK_REGISTRY
import cellstudio.tasks.segmentation
import cellstudio.models.adapters.cellpose_adapter
import cellstudio.pipeline.transforms.loading
import cellstudio.pipeline.transforms.formatting
import cellstudio.pipeline.transforms.visual_aug  # noqa: F401
import traceback

print("1. Loading config...")
cfg = Config.fromfile('configs/segmentation/cellpose_mido_seg.yaml')
cfg.log_config.interval = 1

print("2. Building task...")
try:
    task = TASK_REGISTRY.build({'type': 'InstanceSegmentationTask', 'cfg': cfg})
    
    print("3. Getting runner...")
    task.build_env()
    task.build_model()
    task.build_datasets()
    task.build_evaluator()
    task.build_runner()
    runner = task.runner
    
    print("4. Starting epoch loop...")
    runner.epoch = 0
    runner.call_hook('before_train_epoch')
    print("5. Iterating dataset...")
    
    for i, data_batch in enumerate(runner.train_dataloader):
        print(f"   -> Batch {i} loaded! shape: {data_batch['imgs'].shape}")
        
        imgs = data_batch['imgs'].cuda()
        data_samples = data_batch.get('data_samples')
        
        dtype = next(runner.model.parameters()).dtype
        print(f"   -> Cast imgs to {dtype}...")
        imgs = imgs.to(dtype=dtype)
        
        print("   -> Running self.model(imgs)...")
        with torch.amp.autocast('cuda', enabled=runner.use_amp):
            preds = runner.model.model(imgs)
        print("   -> CNN pass Done.")
        
        print("   -> Extracting masks...")
        mask_list = [ds.gt_instance_seg for ds in data_samples]
            
        print("   -> Generating target flows...")
        target_flows = runner.model._create_flows(mask_list).to(device='cuda', dtype=dtype)
        print("   -> Target flows completed!")
        
        print("   -> Calculating loss...")
        loss = outputs['loss']
        print(f"   -> Backward pass...")
        loss.backward()
        print("   -> Backward pass complete!")
        break
    
    print("All success!")
except Exception as e:
    print(f"CRASH: {e}")
    traceback.print_exc()
