import os
import re
import shutil
import subprocess
import sys
import time

CONFIGS = [
    'configs/detect/yolo_v8m_det_mido_tile.yaml',
    'configs/detect/yolo_26m_det_mido_tile.yaml',
    'configs/detect/faster_rcnn_mido_tile.yaml',
    'configs/detect/detr_mido_tile.yaml',
    'configs/detect/fcos_mido_tile.yaml',
    'configs/detect/rtmdet_mido_tile.yaml',
]

def update_config(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Update max_epochs
    content = re.sub(r'max_epochs:\s*\d+', 'max_epochs: 30', content)
    
    # 2. Update batch_size
    content = re.sub(r'batch_size:\s*\d+', 'batch_size: 4', content)
    
    # 3. Update num_workers
    content = re.sub(r'num_workers:\s*\d+', 'num_workers: 8', content)
    
    # 4. Update milestones and end in MultiStepLR
    if 'detr_mido_tile' in cfg_path:
        content = content.replace('milestones: [100, 130]', 'milestones: [20, 26]')
        content = re.sub(r'(milestones:\s*\[20,\s*26\]\s*\n\s*([^:]+:\s*.*?\n\s*)*?by_epoch:\s*true\s*\n\s*([^:]+:\s*.*?\n\s*)*?begin:\s*\d+\s*\n\s*)end:\s*\d+', r'\g<1>end: 30', content, flags=re.MULTILINE)
        content = re.sub(r'(type:\s*MultiStepLR\s*\n\s*([^:]+:\s*.*?\n\s*)*?begin:\s*\d+\s*\n\s*)end:\s*\d+', r'\g<1>end: 30', content, flags=re.MULTILINE)
    elif 'rtmdet_mido_tile' in cfg_path:
        content = content.replace('milestones: [70, 90]', 'milestones: [21, 27]')
        content = re.sub(r'(type:\s*MultiStepLR\s*\n\s*([^:]+:\s*.*?\n\s*)*?begin:\s*\d+\s*\n\s*)end:\s*\d+', r'\g<1>end: 30', content, flags=re.MULTILINE)
    else:
        content = content.replace('milestones: [40, 80]', 'milestones: [12, 24]')
        # The end parameter might be after begin inside MultiStepLR
        content = re.sub(r'(type:\s*MultiStepLR\s*\n\s*([^:]+:\s*.*?\n\s*)*?begin:\s*\d+\s*\n\s*)end:\s*\d+', r'\g<1>end: 30', content, flags=re.MULTILINE)
        
    with open(cfg_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Updated {cfg_path}: max_epochs=30, batch_size=4, num_workers=8, fixed scheduler")

def clear_work_dir(cfg_path):
    # work_dir is usually 'work_dirs/' + os.path.splitext(os.path.basename(cfg_path))[0]
    name = os.path.splitext(os.path.basename(cfg_path))[0]
    work_dir = os.path.join('work_dirs', name)
    if os.path.exists(work_dir):
        print(f"Clearing work_dir: {work_dir}")
        shutil.rmtree(work_dir)
        # Recreate an empty dir just in case
        os.makedirs(work_dir)

def main():
    # Update all configs first
    for cfg in CONFIGS:
        update_config(cfg)
        
    print("\n" + "="*60)
    print("Starting Training Pipeline")
    print("="*60 + "\n")
    
    # Run all configs one by one
    total = len(CONFIGS)
    for i, cfg in enumerate(CONFIGS):
        name = os.path.splitext(os.path.basename(cfg))[0]
        
        clear_work_dir(cfg)
        
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{total}] TRAINING: {name}")
        print(f"  Config: {cfg}")
        print(f"{'='*60}\n")
        
        t0 = time.time()
        # Launch training process
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

    print(f"\n{'='*60}")
    print(f"  ALL TRAINING COMPLETE")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
