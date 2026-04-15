import os
import json
import random
import shutil
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

def stratified_split_csv(df: pd.DataFrame, label_col: str, ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2), seed: int = 42):
    """6:2:2 stratified split for classification."""
    np.random.seed(seed)
    train_dfs, val_dfs, test_dfs = [], [], []
    
    for label, group in df.groupby(label_col):
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(group)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        
        train_dfs.append(group.iloc[:n_train])
        val_dfs.append(group.iloc[n_train:n_train+n_val])
        test_dfs.append(group.iloc[n_train+n_val:])
        
    return pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs)

def split_custom_json(json_path: str, ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2), seed: int = 42):
    """Split custom format detection dataset into train/val/test."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    items = data['items']
    classes = data['classes']
    
    # Shuffle and split items
    random.seed(seed)
    random.shuffle(items)
    
    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    
    train_items = items[:n_train]
    val_items = items[n_train:n_train+n_val]
    test_items = items[n_train+n_val:]
    
    def build_subset(subset_items):
        return {
            "classes": classes,
            "items": subset_items
        }
        
    return build_subset(train_items), build_subset(val_items), build_subset(test_items)

def process_classification(base_dir: str):
    print("[Splitter] Processing Classification Dataset...")
    csv_path = os.path.join(base_dir, "MIDOG", "MIDOG25_Atypical_Classification_Train_Set.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    # Assuming the label column exists, let's guess its name based on typical MIDOG or print columns
    label_col = 'label' if 'label' in df.columns else df.columns[-1] 
    print(f"Using column '{label_col}' for stratification.")
    
    train_df, val_df, test_df = stratified_split_csv(df, label_col)
    
    out_dir = os.path.join(base_dir, "MIDOG", "splits")
    os.makedirs(out_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"Classification splits saved to {out_dir}")

def process_detection(base_dir: str):
    print("[Splitter] Processing Detection Dataset...")
    json_path = os.path.join(base_dir, "MIDO", "mido_standard.json")
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found.")
        return
        
    train_data, val_data, test_data = split_custom_json(json_path)
    
    out_dir = os.path.join(base_dir, "MIDO", "splits")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "train.json"), 'w') as f:
        json.dump(train_data, f)
    with open(os.path.join(out_dir, "val.json"), 'w') as f:
        json.dump(val_data, f)
    with open(os.path.join(out_dir, "test.json"), 'w') as f:
        json.dump(test_data, f)
    print(f"Detection splits saved to {out_dir}")

def process_segmentation(base_dir: str):
    print("[Splitter] Processing Segmentation Dataset...")
    seg_dir = os.path.join(base_dir, "cellpose", "train")
    if not os.path.exists(seg_dir):
        print(f"Warning: {seg_dir} not found.")
        return
        
    # For cellpose, we typically have image.png and image_masks.png
    files = os.listdir(seg_dir)
    images = [f for f in files if not f.endswith("_masks.png") and f.endswith(".png")]
    
    random.seed(42)
    random.shuffle(images)
    
    n = len(images)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    
    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }
    
    out_dir = os.path.join(base_dir, "cellpose", "splits")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "splits.json"), 'w') as f:
        json.dump(splits, f, indent=4)
        
    print(f"Segmentation splits saved to {out_dir}")

def main():
    base_dir = r"E:\workspace\AlchemyTech\CellStudio\datasets"
    
    print("=== Dataset Splitting (6:2:2) ===")
    process_classification(os.path.join(base_dir, "classfication"))
    process_detection(os.path.join(base_dir, "detection"))
    process_segmentation(os.path.join(base_dir, "segmentation"))
    print("=== Done ===")

if __name__ == "__main__":
    main()
