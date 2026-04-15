import numpy as np
import pytest  # noqa: F401
from pathstudio.utils.wsi_processor import WSIProcessor

def test_wsi_processor_sliding_window():
    processor = WSIProcessor(tile_size=512, overlap=64)
    image_shape = (2000, 3000, 3) # (H, W, C)
    
    tiles = processor.generate_tiles(image_shape)
    
    assert len(tiles) > 0
    # verify grid math: stride is 512-64 = 448
    # w: 3000 / 448 = 6.69 -> 7 cols
    # h: 2000 / 448 = 4.46 -> 5 rows
    # Expected roughly 35 tiles
    assert len(tiles) == 35
    
    # check first tile
    t1 = tiles[0]["box"]
    assert t1 == [0, 0, 512, 512]
    
    # check edge tile constraint
    t_last = tiles[-1]["box"]
    assert t_last[2] == 3000
    assert t_last[3] == 2000
    assert (t_last[2] - t_last[0]) == 512
    assert (t_last[3] - t_last[1]) == 512

def test_wsi_soft_nms():
    WSIProcessor()
    
    boxes = np.array([
        [100, 100, 200, 200], # A
        [105, 105, 205, 205], # B: Highly overlapping with A
        [300, 300, 400, 400]  # C: Distinct
    ], dtype=np.float32)
    scores = np.array([0.9, 0.85, 0.95], dtype=np.float32)
    
    # Method 2: Gaussian Soft-NMS
    keep = WSIProcessor.soft_nms(boxes.copy(), scores.copy(), method=2, threshold=0.5)
    
    # A and C should definitely be kept, B's score will be heavily discounted
    assert len(keep) == 2
    assert 2 in keep # C is kept
    assert 0 in keep # A is kept
