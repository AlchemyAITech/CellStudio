import os
import subprocess
import time
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Test Tiny Pipeline Convergence")
    parser.add_argument('--overfit', action='store_true', help="Run in overfit mode with higher epochs for convergence validation")
    return parser.parse_args()

def main():
    args = parse_args()
    
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_dir = os.path.join(root_dir, 'tests', 'integration', 'configs')
    
    # Define test suite
    # Auto-discover all test configurations (dynamically scale tests to cover YOLO26 and full matrix)
    test_configs = [f for f in os.listdir(config_dir) if f.startswith('tiny_') and f.endswith('.yaml')]
    test_configs.sort()
    
    # Overfit constraints
    pass_thresholds = {
        'cls': 0.95,
        'det': 0.80,
        'seg': 0.70
    }
    
    print("="*60)
    if args.overfit:
        print(">>> Starting CellStudio Convergence Overfit Testing")
    else:
        print("Starting CellStudio Tiny Tests Pipeline")
    print("="*60)
    
    results = []
    
    for cfg_name in test_configs:
        cfg_path = os.path.join(config_dir, cfg_name)
        if not os.path.exists(cfg_path):
            print(f"[SKIPPED]: {cfg_name} not found.")
            continue
            
        print(f"\n[{time.strftime('%H:%M:%S')}] Running Test: {cfg_name}")
        
        work_dir = os.path.join(root_dir, 'work_dirs', 'testing', cfg_name.replace('.yaml', ''))
        os.makedirs(work_dir, exist_ok=True)
        
        # If overfit mode, duplicate config and patch max_epochs
        run_cfg_path = cfg_path
        if args.overfit:
            with open(cfg_path, 'r', encoding='utf8') as f:
                cfg_doc = f.read()

            import re
            cfg_doc = re.sub(r'num_workers:\s*\d+', 'num_workers: 0', cfg_doc)

            # Save in the same dir to preserve relative `_base_` resolution
            run_cfg_path = os.path.join(config_dir, f"overfit_{cfg_name}")
            with open(run_cfg_path, 'w', encoding='utf8') as f:
                f.write(cfg_doc)
                
            target_epochs = 8
            if 'resnet' in cfg_name or 'cls' in cfg_name: target_epochs = 8
            
            with open(run_cfg_path, 'a', encoding='utf8') as f:
                f.write(f"\n\nrunner:\n  max_epochs: {target_epochs}\n")
                f.write(f"\ndefault_hooks:\n  checkpoint_hook:\n    interval: 100\n    save_best: null\n")
                
        cmd = [
            'python', 'tools/train.py',
            run_cfg_path,
            '--work-dir', work_dir
        ]
        
        start_time = time.time()
        try:
            # Capture output for metrics parsing
            res = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True, encoding='utf8', errors='ignore')
            duration = time.time() - start_time
            
            if res.returncode == 0:
                print(f"[PASSED]: {cfg_name} (in {duration:.1f}s)")
                
                # Metric Validation
                if args.overfit:
                    log_output = res.stdout + res.stderr
                    metric_status = "UNKNOWN"
                    final_loss = 999.0
                    
                    import re
                    # Look for loss: 0.xxx or val_loss: 0.xxx
                    loss_matches = re.findall(r'loss(?:_cls)?[\s:]*([\d\.]+)', log_output)
                    if not loss_matches:
                        loss_matches = re.findall(r'val_loss[\s:]*([\d\.]+)', log_output)
                        
                    metric_status = "UNKNOWN"
                    passed = False
                    
                    if loss_matches:
                        final_loss = float(loss_matches[-1])
                        metric_status = f"LOSS={final_loss:.3f}"
                        target_thresh = 0.5
                        if 'yolov8' in cfg_name or 'faster_rcnn' in cfg_name or 'yolo26' in cfg_name:
                             target_thresh = 0.65
                        if 'resnet50' in cfg_name:
                             target_thresh = 0.3
                        passed = final_loss <= target_thresh
                    
                    # Target Metric Parsing
                    acc_match = re.search(r'(?:accuracy|acc)[\s:=]*([\d\.]+)', log_output, re.IGNORECASE)
                    map_match = re.search(r'(?:mAP|coco/bbox_mAP)[\s:=]*([\d\.]+)', log_output, re.IGNORECASE)
                    miou_match = re.search(r'(?:mIoU|MeanIoU)[\s:=]*([\d\.]+)', log_output, re.IGNORECASE)
                    
                    if map_match:
                        metric_val = float(map_match.group(1))
                        metric_status = f"mAP={metric_val:.3f}"
                        passed = metric_val >= 0.80 # Expert expectation for tiny overfit
                    elif miou_match:
                        metric_val = float(miou_match.group(1))
                        metric_status = f"mIoU={metric_val:.3f}"
                        passed = metric_val >= 0.70
                    elif acc_match:
                        metric_val = float(acc_match.group(1))
                        if metric_val > 1.0: metric_val /= 100.0
                        metric_status = f"Acc={metric_val:.3f}"
                        passed = metric_val >= 0.95
                        
                    if passed:
                        results.append((cfg_name, f"CONVERGED ({metric_status})", duration))
                    else:
                        results.append((cfg_name, f"FAILED_QC ({metric_status})", duration))
                        print(f"  --> Quality Control Failed! {metric_status} did not meet criteria.")
                else:
                    results.append((cfg_name, "PASSED", duration))
            else:
                print(f"[FAILED]: {cfg_name} (Exit Code: {res.returncode})")
                print(res.stderr[-1000:]) # print last 1000 chars of err
                results.append((cfg_name, "FAILED", duration))
        except Exception as e:
            duration = time.time() - start_time
            print(f"[CRASHED]: {cfg_name} ({e})")
            results.append((cfg_name, "CRASHED", duration))

    # Print summary
    print("\n" + "="*60)
    print("Test Suite Summary")
    print("="*60)
    
    all_passed = True
    for name, status, dur in results:
        status_sym = "[v]" if "PASSED" in status or "CONVERGED" in status or "RUN_OK" in status else "[x]"
        all_passed = all_passed and (status_sym == "[v]")
        # Pad strings for nice table output
        print(f"{status_sym} {name:<25} | {status:<25} | {dur:4.1f}s")
        
    print("="*60)
    if all_passed:
        print("[SUCCESS] ALL TESTS VALIDATED SUCCESSFULLY.")
    else:
        print("[WARNING] SOME TESTS FAILED OR DID NOT CONVERGE. CHECK LOGS ABOVE.")
        
if __name__ == '__main__':
    main()
