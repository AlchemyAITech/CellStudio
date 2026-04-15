"""
Batch tile-based training for all 6 detection models.
Run sequentially since they share a single GPU.
"""
import subprocess
import sys
import os
import time

CONFIGS = [
    'configs/detect/yolo_v8m_det_mido_tile.yaml',
    'configs/detect/yolo_26m_det_mido_tile.yaml',
    'configs/detect/faster_rcnn_mido_tile.yaml',
    'configs/detect/detr_mido_tile.yaml',
    'configs/detect/fcos_mido_tile.yaml',
    'configs/detect/rtmdet_mido_tile.yaml',
]

def main():
    total = len(CONFIGS)
    for i, cfg in enumerate(CONFIGS):
        name = os.path.splitext(os.path.basename(cfg))[0]
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{total}] TRAINING: {name}")
        print(f"  Config: {cfg}")
        print(f"{'='*60}\n")
        
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, 'tools/train.py', cfg],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        )
        elapsed = time.time() - t0
        
        status = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
        print(f"\n  [{i+1}/{total}] {name}: {status} ({elapsed/60:.1f} min)")
    
    print(f"\n{'='*60}")
    print(f"  ALL TRAINING COMPLETE")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
