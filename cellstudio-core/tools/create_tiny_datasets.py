"""Create tiny test datasets from full MIDOG/MIDO/Cellpose datasets.

Tiny dataset specs:
  - Classification (MIDOG): 120 images → train/val=100, test=20
  - Detection (MIDO):        12 images → train/val=10,  test=2
  - Segmentation (Cellpose):  60 images → train/val=50, test=10

Usage:
    python tools/create_tiny_datasets.py
"""

import csv
import json
import os
import random
import shutil

random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# 1. Classification — MIDOG tiny
# ============================================================================
def create_cls_tiny():
    src_img = os.path.join(ROOT, "datasets", "classfication", "MIDOG",
                           "MIDOG25_Binary_Classification_Train_Set")
    src_split = os.path.join(ROOT, "datasets", "classfication", "MIDOG", "splits")
    dst_root = os.path.join(ROOT, "datasets", "classfication", "MIDOG_tiny")

    # Read existing split CSVs — schema:
    #   image_id, filename, coordinateX, coordinateY, Tumor, Tissue, Species,
    #   expert1, expert2, expert3, majority
    all_samples = []  # list of full row dicts
    fieldnames = None

    for csv_name in ["train.csv", "val.csv", "test.csv"]:
        csv_path = os.path.join(src_split, csv_name)
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                all_samples.append(row)

    print(f"[CLS] Total samples: {len(all_samples)}")

    # Group by majority label for balanced sampling
    by_label = {}
    for row in all_samples:
        lbl = row.get("majority", "unknown")
        by_label.setdefault(lbl, []).append(row)

    for lbl, items in by_label.items():
        print(f"  label={lbl}: {len(items)} samples")

    # Sample 60 per class (120 total for 2-class binary: AMF / NMF)
    n_labels = len(by_label)
    per_class = 120 // n_labels
    per_class_train = 100 // n_labels
    per_class_test = per_class - per_class_train

    train_val_rows = []
    test_rows = []
    for lbl, items in by_label.items():
        random.shuffle(items)
        train_val_rows.extend(items[:per_class_train])
        test_rows.extend(items[per_class_train:per_class])

    # Create directories (idempotent)
    img_dir = os.path.join(dst_root, "images")
    split_dir = os.path.join(dst_root, "splits")
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    # Copy images
    copied = 0
    for row in train_val_rows + test_rows:
        fname = row["image_id"]  # e.g. "11826.png"
        src = os.path.join(src_img, fname)
        dst = os.path.join(img_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1

    # Write split CSVs (same schema as original)
    for split_name, rows in [("train", train_val_rows),
                              ("val", train_val_rows),
                              ("test", test_rows)]:
        csv_path = os.path.join(split_dir, f"{split_name}.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"[CLS] Created MIDOG_tiny: {copied} images, "
          f"train/val={len(train_val_rows)}, test={len(test_rows)}")


# ============================================================================
# 2. Detection — MIDO tiny
# ============================================================================
def create_det_tiny():
    src_dir = os.path.join(ROOT, "datasets", "detection", "MIDO")
    src_split = os.path.join(src_dir, "splits")
    dst_root = os.path.join(ROOT, "datasets", "detection", "MIDO_tiny")

    # Read train/test splits which have schema: { "classes": [...], "items": [...] }
    train_json = os.path.join(src_split, "train.json")
    with open(train_json, "r", encoding="utf-8") as f:
        train_split = json.load(f)
    
    test_json = os.path.join(src_split, "test.json")
    with open(test_json, "r", encoding="utf-8") as f:
        test_split = json.load(f)

    classes = train_split.get("classes", [])
    
    def get_available_items(items):
        avail = []
        for item in items:
            fname = os.path.basename(item["image_path"])
            src = os.path.join(src_dir, fname)
            if os.path.exists(src):
                avail.append(item)
        return avail

    train_items = get_available_items(train_split.get("items", []))
    test_items = get_available_items(test_split.get("items", []))

    print(f"[DET] Total items (available): train={len(train_items)}, test={len(test_items)}")

    # Shuffle and select items
    random.shuffle(train_items)
    train_val_imgs = train_items[:10]

    random.shuffle(test_items)
    if len(test_items) >= 2:
        test_imgs = test_items[:2]
    else:
        # fallback
        test_imgs = train_items[10:12]

    # Create directories (idempotent)
    split_dir = os.path.join(dst_root, "splits")
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(dst_root, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    # Copy / symlink images based on image_path
    def process_items(items):
        processed = []
        for item in items:
            # item["image_path"] might be absolute or relative, extract fname
            fname = os.path.basename(item["image_path"])
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_root, fname)
            
            if os.path.exists(src):
                if not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
            
                # Update image_path to the new location (or keep as relative, we'll store relative just in case, but let's stick to original logic: just basename if we prefer, but for YOLO compatibility, we might need absolute or what it had)
                # We'll just replace with new absolute path for safety, or keep basename
                new_item = item.copy()
                new_item["image_path"] = dst
                processed.append(new_item)
        return processed

    train_val_processed = process_items(train_val_imgs)
    test_processed = process_items(test_imgs)

    for split_name, item_list in [("train", train_val_processed),
                                  ("val", train_val_processed),
                                  ("test", test_processed)]:
        out_json = {
            "classes": classes,
            "items": item_list
        }
        path = os.path.join(split_dir, f"{split_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)

    # Copy the main custom mido_standard.json
    anno_path = os.path.join(src_dir, "mido_standard.json")
    if os.path.exists(anno_path):
        shutil.copy2(anno_path, os.path.join(dst_root, "mido_standard.json"))

    print(f"[DET] Created MIDO_tiny: train/val={len(train_val_processed)}, "
          f"test={len(test_processed)}")


# ============================================================================
# 3. Segmentation — Cellpose tiny
# ============================================================================
def create_seg_tiny():
    src_train = os.path.join(ROOT, "datasets", "segmentation", "cellpose", "train")
    src_test = os.path.join(ROOT, "datasets", "segmentation", "cellpose", "test")
    dst_root = os.path.join(ROOT, "datasets", "segmentation", "cellpose_tiny")

    # Enumerate available image IDs
    train_ids = sorted(set(
        f.replace("_img.png", "")
        for f in os.listdir(src_train)
        if f.endswith("_img.png")
    ))

    test_ids = sorted(set(
        f.replace("_img.png", "")
        for f in os.listdir(src_test)
        if f.endswith("_img.png")
    )) if os.path.exists(src_test) else []

    print(f"[SEG] Train images available: {len(train_ids)}, "
          f"Test images available: {len(test_ids)}")

    # Sample 50 from train for train/val, 10 from test for test
    random.shuffle(train_ids)
    train_val_ids = train_ids[:50]

    if len(test_ids) >= 10:
        random.shuffle(test_ids)
        test_sel_ids = test_ids[:10]
        test_src = src_test
    else:
        test_sel_ids = train_ids[50:60]
        test_src = src_train

    # Create output (idempotent)
    dst_train = os.path.join(dst_root, "train")
    dst_test = os.path.join(dst_root, "test")
    dst_splits = os.path.join(dst_root, "splits")
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_test, exist_ok=True)
    os.makedirs(dst_splits, exist_ok=True)

    for img_id in train_val_ids:
        for suffix in ["_img.png", "_masks.png"]:
            src = os.path.join(src_train, f"{img_id}{suffix}")
            dst = os.path.join(dst_train, f"{img_id}{suffix}")
            if os.path.exists(src):
                shutil.copy2(src, dst)

    for img_id in test_sel_ids:
        for suffix in ["_img.png", "_masks.png"]:
            src = os.path.join(test_src, f"{img_id}{suffix}")
            dst = os.path.join(dst_test, f"{img_id}{suffix}")
            if os.path.exists(src):
                shutil.copy2(src, dst)

    for split_name, ids in [("train", train_val_ids),
                             ("val", train_val_ids),
                             ("test", test_sel_ids)]:
        path = os.path.join(dst_splits, f"{split_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2)

    print(f"[SEG] Created cellpose_tiny: train/val={len(train_val_ids)}, "
          f"test={len(test_sel_ids)}")


# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Creating Tiny Datasets for CellStudio Testing")
    print("=" * 60)

    print("\n--- Classification (MIDOG) ---")
    create_cls_tiny()

    print("\n--- Detection (MIDO) ---")
    create_det_tiny()

    print("\n--- Segmentation (Cellpose) ---")
    create_seg_tiny()

    print("\n" + "=" * 60)
    print("All tiny datasets created successfully!")
    print("=" * 60)
