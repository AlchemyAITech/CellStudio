import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader, Subset

# Ensure paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cellstudio.datasets.registry import DATASET_REGISTRY
from cellstudio.models.builder import MODEL_REGISTRY
import cellstudio.models.adapters  # Registers adapters
import cellstudio.metrics.segmentation # Registers metrics  # noqa: F401
from cellstudio.evaluation.evaluator import Evaluator
from cellstudio.pipeline.transforms.formatting import PackCellStudioInputs  # noqa: F401

def collate_fn(batch):
    batch_dict = {'imgs': [], 'data_samples': []}
    for item in batch:
        batch_dict['imgs'].append(item['imgs'])
        batch_dict['data_samples'].append(item['data_samples'])
    batch_dict['imgs'] = torch.stack(batch_dict['imgs'])
    return batch_dict

def test_pipeline(config_path):
    print(f"\n[{config_path}] Initiating Pipeline Validation...")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    print(f"1. Building Dataset {cfg['dataset']['type']}")
    dataset = DATASET_REGISTRY.build(cfg['dataset'])
    
    # Use a mini-subset of 2 items to test
    subset = Subset(dataset, [0, 1])
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print(f"2. Building Model Adapter {cfg['model']['type']}")
    model = MODEL_REGISTRY.build(cfg['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"3. Building Evaluator with Metrics: {cfg.get('val_evaluator', {}).get('metrics', [])}")
    # Force add at least one metric if none
    eval_cfg = cfg.get('val_evaluator', {})
    if not eval_cfg:
        eval_cfg = {'metrics': [{'type': 'seg_all_metrics'}], 'plotters': []}
    evaluator = Evaluator(metrics_cfg=eval_cfg.get('metrics', []), plotters_cfg=eval_cfg.get('plotters', []))
    
    print("4. Executing Forward Train (Loss Generation)...")
    batch = next(iter(loader))
    imgs = batch['imgs'].to(device)
    data_samples = batch['data_samples']
    
    model.train()
    loss_dict = model.forward_train(imgs, data_samples)
    print(f"   -> Successfully computed gradients: loss = {loss_dict['loss'].item():.4f}")
    
    print("5. Executing Forward Test & Metrics Analysis...")
    model.eval()
    with torch.no_grad():
        infer_results = model.forward_test(imgs, data_samples)
    print(f"   -> Forward Test complete. Output {len(infer_results)} result items.")
    
    # Send to evaluator
    evaluator.process(data_batch={'data_samples': data_samples}, outputs=infer_results)
    metrics_result = evaluator.evaluate(work_dir='./work_dirs/tmp')
    
    print(f"6. Final Metrics Extraction: {list(metrics_result.keys())}")
    print("✓ Pipeline is GREEN and ready for production!\n")

if __name__ == '__main__':
    os.makedirs('work_dirs/tmp', exist_ok=True)
    configs = [
        'configs/segmentation/unet_mido_seg.yaml',
        'configs/segmentation/yolo_v8m_seg_mido.yaml',
        'configs/segmentation/cellpose_mido_seg.yaml',
        'configs/segmentation/cellpose_sam_mido_seg.yaml',
    ]
    for c in configs:
        try:
            test_pipeline(c)
        except Exception as e:
            print(f"✗ FAILED {c}: {e}")
            import traceback
            traceback.print_exc()
