"""Debug paths for TileMIDODataset."""
import os, json, sys  # noqa: F401

data_root = 'datasets/detection/MIDO'
print(f"CWD: {os.getcwd()}")
print(f"data_root: {data_root}")
print(f"data_root abs: {os.path.abspath(data_root)}")

for split in ['train', 'val']:
    ann_path = os.path.join(data_root, 'annotations', f'{split}.json')
    ann = json.load(open(ann_path, encoding='utf-8'))
    print(f"\n--- {split} ---")
    print(f"Items: {len(ann['items'])}")
    
    # Check first and a random item
    for i in [0, len(ann['items'])//2]:
        item = ann['items'][i]
        raw_path = item['image_path']
        basename = os.path.basename(raw_path)
        constructed = os.path.join(data_root, basename)
        exists = os.path.exists(constructed)
        print(f"  [{i}] raw={raw_path}, basename={basename}, path={constructed}, exists={exists}")

# Specifically check 158.tiff
p158 = os.path.join(data_root, '158.tiff')
print(f"\n158.tiff exists: {os.path.exists(p158)}")

# List actual files in data_root
files = [f for f in os.listdir(data_root) if f.endswith(('.tiff', '.tif', '.png', '.jpg'))]
print(f"Image files in data_root: {len(files)}")
if files:
    print(f"  First 5: {files[:5]}")
    # Check if 158 exists with any extension
    f158 = [f for f in files if '158' in f]
    print(f"  Files matching '158': {f158}")
