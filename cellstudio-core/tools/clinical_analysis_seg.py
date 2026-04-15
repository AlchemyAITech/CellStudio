import os
import sys  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from cellstudio.engine.config.config import Config
from cellstudio.tasks.registry import TASK_REGISTRY
from cellstudio.datasets.registry import DATASET_REGISTRY  # noqa: F401

# Required imports
import cellstudio.datasets.segmentation
import cellstudio.pipeline.transforms.loading
import cellstudio.pipeline.transforms.formatting
import cellstudio.pipeline.transforms.visual_aug  # noqa: F401

CONFIGS = {
    'UNet': 'configs/segmentation/unet_mido_seg.yaml',
    'DeepLabV3': 'configs/segmentation/deeplabv3_mido_seg.yaml',
    'YOLOv8-Seg': 'configs/segmentation/yolo_v8m_seg_mido.yaml',
    'CellposeV3': 'configs/segmentation/cellpose_mido_seg.yaml',
    'Cellpose-SAM': 'configs/segmentation/cellpose_sam_mido_seg.yaml'
}

def analyze_counts_for_model(model_name, cfg_path):
    print(f"\nEvaluating Model: {model_name}")
    cfg = Config.fromfile(cfg_path)
    work_dir = os.path.join('work_dirs', os.path.splitext(os.path.basename(cfg_path))[0])
    weights_path = os.path.join(work_dir, 'best.pth')
    
    if not os.path.exists(weights_path):
        weights_path = os.path.join(work_dir, 'latest.pth')
        if not os.path.exists(weights_path):
            print(f"Skipping {model_name}: No checkpoint found.")
            return None
        
    print(f"Loading checkpoint: {weights_path}")
    
    # Init Task to build runner and models securely
    task = TASK_REGISTRY.build({'type': 'InstanceSegmentationTask', 'cfg': cfg})
    task.build_model()
    task.build_datasets()
    task.model.load_state_dict(torch.load(weights_path, map_location='cuda'))
    task.model.eval()
    
    dataloader = task.val_dataloader
    
    gt_counts = []
    pred_counts = []
    
    print("Running Inference...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            imgs = batch['imgs'].cuda()
            dtype = next(task.model.parameters()).dtype
            imgs = imgs.to(dtype=dtype)
            
            data_samples = batch.get('data_samples', [])
            
            # Use forward_test directly
            if hasattr(task.model, 'forward_test'):
                results = task.model.forward_test(imgs, data_samples)
            else:
                results = task.model(imgs, mode='predict', data_samples=data_samples)
                
            for res, ds in zip(results, data_samples):
                # Count GT
                gt_mask = ds.gt_instance_seg
                if torch.is_tensor(gt_mask): gt_mask = gt_mask.cpu().numpy()
                num_gt = len([idx for idx in np.unique(gt_mask) if idx != 0])
                gt_counts.append(num_gt)
                
                # Count Pred
                pred_mask = res.masks
                if torch.is_tensor(pred_mask): 
                    # N, H, W
                    num_pred = pred_mask.shape[0] if len(pred_mask.shape) == 3 else len(np.unique(pred_mask.cpu().numpy())) - 1
                else:
                    num_pred = len([idx for idx in np.unique(pred_mask) if idx != 0])
                
                # Minimum bounds check
                pred_counts.append(max(0, num_pred))
                
    return gt_counts, pred_counts

def make_bland_altman(data_df):
    plt.figure(figsize=(15, 10))
    models = [c for c in data_df.columns if c != 'GT']
    
    for i, model in enumerate(models):
        plt.subplot(2, 3, i+1)
        m1 = data_df['GT']
        m2 = data_df[model]
        
        mean_vals = (m1 + m2) / 2.0
        diff_vals = m2 - m1
        md = np.mean(diff_vals)
        sd = np.std(diff_vals, axis=0)
        
        plt.scatter(mean_vals, diff_vals, alpha=0.5, edgecolor='black', s=20)
        plt.axhline(md, color='red', linestyle='-', linewidth=2)
        plt.axhline(md + 1.96*sd, color='blue', linestyle='--', linewidth=2)
        plt.axhline(md - 1.96*sd, color='blue', linestyle='--', linewidth=2)
        
        plt.title(f"{model} Bland-Altman")
        plt.xlabel('Mean of Pred and GT')
        plt.ylabel('Difference (Pred - GT)')
        
    plt.tight_layout()
    plt.savefig('bland_altman_segmentation.png', dpi=200)
    print("Saved Bland-Altman plot to 'bland_altman_segmentation.png'")

def main():
    print("Extracting Instance Validation Counts...")
    df_data = {}
    
    for k, v in CONFIGS.items():
        res = analyze_counts_for_model(k, v)
        if res:
            gt, pred = res
            if 'GT' not in df_data:
                df_data['GT'] = gt
            df_data[k] = pred
            
    if len(df_data) <= 1:
        print("Not enough complete testing data to plot. Exiting.")
        return
        
    df = pd.DataFrame(df_data)
    
    # ICC / Correlation Analysis
    print("\nIntraclass Correlation Pass (Pearson/Spearman approximation):")
    from scipy.stats import pearsonr, spearmanr
    for model in [c for c in df.columns if c != 'GT']:
        p_val, _ = pearsonr(df['GT'], df[model])
        s_val, _ = spearmanr(df['GT'], df[model])
        print(f"  {model} vs GT -> Pearson R: {p_val:.4f} | Spearman Rho: {s_val:.4f}")
        
    make_bland_altman(df)

if __name__ == '__main__':
    main()
