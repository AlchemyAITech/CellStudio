"""Update tile configs: 2048x2048 crop -> resize 1024x1024"""
import os  # noqa: F401

configs = [
    'configs/detect/yolo_v8m_det_mido_tile.yaml',
    'configs/detect/yolo_26m_det_mido_tile.yaml',
    'configs/detect/faster_rcnn_mido_tile.yaml',
    'configs/detect/detr_mido_tile.yaml',
    'configs/detect/fcos_mido_tile.yaml',
    'configs/detect/rtmdet_mido_tile.yaml',
]

for cfg_path in configs:
    with open(cfg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Train dataset: tile_size 1024 -> 2048, add resize_to, stride 1024 -> 2048
    content = content.replace(
        '    tile_size: [1024, 1024]\n    stride: 1024\n    min_boxes: 1',
        '    tile_size: [2048, 2048]\n    resize_to: [1024, 1024]\n    stride: 2048\n    min_boxes: 1'
    )
    
    # Val dataset: tile_size 1024 -> 2048, add resize_to, stride stays 2048
    content = content.replace(
        '    tile_size: [1024, 1024]\n    stride: 2048\n    min_boxes: 0',
        '    tile_size: [2048, 2048]\n    resize_to: [1024, 1024]\n    stride: 2048\n    min_boxes: 0'
    )
    
    if content != original:
        with open(cfg_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {cfg_path}")
    else:
        print(f"Unchanged: {cfg_path}")
