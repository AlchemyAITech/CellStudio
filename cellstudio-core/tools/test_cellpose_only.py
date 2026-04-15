import os
import shutil
import subprocess
import sys
import time
import glob  # noqa: F401

CONFIGS = [
    'configs/segmentation/cellpose_mido_seg.yaml',
    'configs/segmentation/cellpose_sam_mido_seg.yaml'
]

def clear_work_dir(cfg_path):
    name = os.path.splitext(os.path.basename(cfg_path))[0]
    work_dir = os.path.join('work_dirs', name)
    if os.path.exists(work_dir):
        print(f"Clearing work_dir: {work_dir}")
        shutil.rmtree(work_dir)
        os.makedirs(work_dir)

def main():
    print("\n" + "="*60)
    print("Starting targeted testing: Cellpose & Cellpose-SAM")
    print("="*60 + "\n")
    
    total = len(CONFIGS)
    for i, cfg in enumerate(CONFIGS):
        name = os.path.splitext(os.path.basename(cfg))[0]
        clear_work_dir(cfg)
        
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
        
        if result.returncode != 0:
            print(f"Stopping pipeline due to failure in {name}.")
            break

if __name__ == '__main__':
    main()
