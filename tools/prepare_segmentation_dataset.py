import os
import sys
import json  # noqa: F401
import shutil
import random

# Add path so we can import cellstudio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cellstudio.datasets.schema import CellDatasetConfig

def clean_extracted_folders(base_dir):
    """Move files from train/train/* to train/* and remove empty dir."""
    for split in ['train', 'test']:
        nested_dir = os.path.join(base_dir, split, split)
        target_dir = os.path.join(base_dir, split)
        
        if os.path.exists(nested_dir):
            print(f"Fixing nested directory {nested_dir} -> {target_dir}")
            for filename in os.listdir(nested_dir):
                src = os.path.join(nested_dir, filename)
                dst = os.path.join(target_dir, filename)
                shutil.move(src, dst)
            os.rmdir(nested_dir)

def prepare_dataset():
    base_dir = "datasets/segmentation/cellpose"
    json_path = os.path.join(base_dir, "cellpose_standard.json")
    
    clean_extracted_folders(base_dir)
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
        
    print(f"Loading {json_path}...")
    dataset = CellDatasetConfig.load(json_path)
    
    # Clean paths
    valid_items = []
    print("Validating all item paths...")
    for item in dataset.items:
        # Expected e.g., /Users/.../datasets/segmentation/cellpose/train/164_img.png
        p = item.image_path.replace('\\', '/')
        if 'cellpose/' in p:
            rel_path = p.split('cellpose/')[-1]
        else:
            rel_path = os.path.basename(p)
            
        # Also clean up mask_path if it exists
        if item.mask_path:
            mp = item.mask_path.replace('\\', '/')
            if 'cellpose/' in mp:
                rel_mask_path = mp.split('cellpose/')[-1]
            else:
                rel_mask_path = os.path.basename(mp)
            item.mask_path = rel_mask_path
            
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            item.image_path = rel_path
            valid_items.append(item)
        else:
            print(f"WARNING: File missing - {full_path}")
            
    print(f"Successfully validated {len(valid_items)}/{len(dataset.items)} items.")
    
    # 5-Fold split Generation
    k_folds = 5
    random.seed(42)
    random.shuffle(valid_items)
    
    splits_dir = os.path.join(base_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    fold_size = len(valid_items) // k_folds
    
    for i in range(k_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k_folds - 1 else len(valid_items)
        
        val_items = valid_items[val_start:val_end]
        train_items = valid_items[:val_start] + valid_items[val_end:]
        
        train_config = CellDatasetConfig(items=train_items, classes=dataset.classes)
        val_config = CellDatasetConfig(items=val_items, classes=dataset.classes)
        
        train_file = os.path.join(splits_dir, f"train_fold{i}.json")
        val_file = os.path.join(splits_dir, f"val_fold{i}.json")
        
        train_config.save(train_file)
        val_config.save(val_file)
        print(f"Created Fold {i}: Train ({len(train_items)}), Val ({len(val_items)})")
        
    print("Dataset preparation complete.")

if __name__ == '__main__':
    prepare_dataset()
