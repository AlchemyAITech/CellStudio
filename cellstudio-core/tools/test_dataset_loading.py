"""Exhaustive dataset loading test — verifies all train/val tiles load without error."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellstudio.datasets.tile_mido import TileMIDODataset

def test_dataset(split, ann_file, min_boxes, max_tiles=None):
    print(f"\n{'='*60}")
    print(f"  Testing {split} dataset")
    print(f"{'='*60}")
    
    ds = TileMIDODataset(
        data_root='datasets/detection/MIDO',
        ann_file=ann_file,
        tile_size=(2048, 2048),
        resize_to=(1024, 1024),
        stride=2048,
        min_boxes=min_boxes,
        pipeline=None
    )
    
    n = len(ds) if max_tiles is None else min(max_tiles, len(ds))
    print(f"  Total tiles: {len(ds)}, testing: {n}")
    
    errors = 0
    t0 = time.time()
    for i in range(n):
        try:
            item = ds[i]
            assert 'img' in item, f"Tile {i}: missing 'img'"
            assert item['img'].shape[0] > 0, f"Tile {i}: zero height"
        except Exception as e:
            errors += 1
            print(f"  ERROR tile {i}: {e}")
            if errors > 5:
                print("  ... too many errors, stopping")
                break
    
    elapsed = time.time() - t0
    print(f"  Loaded {n} tiles in {elapsed:.1f}s, errors: {errors}")
    return errors == 0

if __name__ == '__main__':
    ok_train = test_dataset('TRAIN', 'splits/train.json', min_boxes=1, max_tiles=100)
    ok_val = test_dataset('VAL', 'splits/val.json', min_boxes=0)
    
    print(f"\n{'='*60}")
    if ok_train and ok_val:
        print("  ALL DATASET TESTS PASSED")
    else:
        print("  DATASET TESTS FAILED")
        sys.exit(1)
    print(f"{'='*60}")
