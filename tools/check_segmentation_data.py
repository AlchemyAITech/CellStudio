import os
import sys

# Add path so we can import cellstudio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cellstudio.datasets.schema import CellDatasetConfig

def main():
    json_path = "datasets/segmentation/cellpose/cellpose_standard.json"
    print(f"Loading {json_path}...")
    dataset = CellDatasetConfig.load(json_path)
    
    print(f"Total items: {len(dataset.items)}")
    print(f"Classes: {dataset.classes}")
    
    if dataset.items:
        first = dataset.items[0]
        print("\nFirst Item Info:")
        print(f"Image path: {first.image_path}")
        print(f"Image WxH: {first.image_width}x{first.image_height}")
        print(f"Number of polygons: {len(first.polygons)}")
        
        # Check path fixes
        fixed_path = first.image_path
        if 'cellpose/' in fixed_path:
            fixed_path = fixed_path.split('cellpose/')[-1]
        
        full_path = os.path.join("datasets/segmentation/cellpose", fixed_path)
        print(f"Fixed relative path: {fixed_path}")
        print(f"Exists on disk? {os.path.exists(full_path)} ({full_path})")
        
        # Calculate polygon stats
        num_polys = [len(item.polygons) for item in dataset.items]
        print(f"\nAvg polygons per image: {sum(num_polys) / len(num_polys):.2f}")
        print(f"Max polygons in an image: {max(num_polys)}")

if __name__ == '__main__':
    main()
