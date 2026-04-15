import os
import json
import matplotlib.pyplot as plt
import numpy as np

WORK_DIRS = [
    'work_dirs/unet_mido_seg',
    'work_dirs/deeplabv3_mido_seg',
    'work_dirs/yolo_v8m_seg_mido',
    'work_dirs/cellpose_mido_seg',
    'work_dirs/cellpose_sam_mido_seg'
]

def main():
    print("Parsing Zenith Segmentation Scalars...")
    all_metrics = {}
    
    for wd in WORK_DIRS:
        name = os.path.basename(wd).replace('_mido_seg', '').replace('_seg_mido', '')
        scalar_file = os.path.join(wd, 'scalars.json')
        
        if not os.path.exists(scalar_file):
            print(f"Skipping {name}, no scalars.json found yet.")
            continue
            
        best_metrics = {}
        best_score = -1
        
        with open(scalar_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                if record.get('mode') == 'val':
                    # We look for the best F1 or mIoU
                    score = record.get('PQ', 0.0) + record.get('F1', 0.0)
                    if score > best_score:
                        best_score = score
                        best_metrics = record
                        
        if best_metrics:
            all_metrics[name] = best_metrics
            
    if not all_metrics:
        print("No validation metrics available yet to parse.")
        return
        
    print("\n--- Model Comparison Summary ---")
    headers = ['Model', 'Epoch', 'PQ', 'mIoU', 'Dice', 'F1', 'HD95', 'Count_MAE', 'AJI']
    print(f"{headers[0]:<15} | {headers[1]:<5} | {headers[2]:<6} | {headers[3]:<6} | {headers[4]:<6} | {headers[5]:<6} | {headers[6]:<6} | {headers[7]:<9} | {headers[8]:<6}")
    print("-" * 80)
    for model, m in all_metrics.items():
        print(f"{model:<15} | {m.get('epoch', 0):<5} | {m.get('PQ', 0):.4f} | {m.get('mIoU', 0):.4f} | {m.get('Dice', 0):.4f} | {m.get('F1', 0):.4f} | {m.get('HD95', 0):.2f} | {m.get('Count_MAE', 0):.2f}    | {m.get('AJI', 0):.4f}")
        
    # Create Bar Chart
    models = list(all_metrics.keys())
    pqs = [all_metrics[m].get('PQ', 0) for m in models]
    f1s = [all_metrics[m].get('F1', 0) for m in models]
    mious = [all_metrics[m].get('mIoU', 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, pqs, width, label='PQ (Panoptic Quality)')
    plt.bar(x, f1s, width, label='F1 (Cell Discovery)')
    plt.bar(x + width, mious, width, label='mIoU (Pixel Match)')
    
    plt.ylabel('Score')
    plt.title('Segmentation Model Performance Comparison (MIDOG)')
    plt.xticks(x, models, rotation=15)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('segmentation_comparison.png', dpi=150)
    print("\nSaved chart to segmentation_comparison.png")

if __name__ == '__main__':
    main()
