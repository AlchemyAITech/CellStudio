import os
import shutil
import subprocess
import time

ordered_configs = [
    "configs/segmentation/cellpose_mido_seg.yaml",
    "configs/segmentation/cellpose_sam_mido_seg.yaml",
    "configs/segmentation/unet_mido_seg.yaml",
    "configs/segmentation/deeplabv3_mido_seg.yaml",
    "configs/segmentation/yolo_v8s_seg_mido.yaml"
]

def clear_work_dir(cfg_path):
    name = os.path.splitext(os.path.basename(cfg_path))[0]
    work_dir = os.path.join('work_dirs', name)
    if os.path.exists(work_dir):
        print(f"Clearing work_dir: {work_dir}")
        shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

def main():
    print("=======================================")
    print(" ZENITH ORDERED 5-MODEL LAUNCHER")
    print("=======================================")
    
    # 1. Clear ALL work directories first guarantees clean starts!
    for cfg in ordered_configs:
        clear_work_dir(cfg)
        
    print("\n[+] All work directories cleared. Commencing ordered execution...\n")
    
    # 2. Run them strictly in order
    for idx, cfg in enumerate(ordered_configs):
        print(f"[{idx+1}/{len(ordered_configs)}] 🚀 LAUNCHING: {cfg}")
        start_time = time.time()
        
        # Execute training inline
        cmd = ["python", "tools/train.py", cfg]
        result = subprocess.run(cmd)
        
        elapsed = time.time() - start_time
        if result.returncode != 0:
            print(f"❌ ERROR: Subprocess crashed with strictly non-zero code {result.returncode} for {cfg}!")
            print("Terminating sequential launch sequence.")
            break
        else:
            print(f"✅ SUCCESS: Finished {cfg} in {elapsed/60:.2f} minutes.\n")
            
    print("=======================================")
    print("ALL SCHEDULED TRAININGS CONCLUDED.")

if __name__ == '__main__':
    main()
