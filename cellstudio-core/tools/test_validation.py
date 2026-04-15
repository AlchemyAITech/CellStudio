"""Quick 1-epoch test: verify train + val completes without crash."""
import sys, os, shutil, tempfile, yaml  # noqa: F401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Registry imports (same as train.py)
from cellstudio.engine.config.config import Config
from cellstudio.tasks.registry import TASK_REGISTRY
import cellstudio.tasks.object_detection
import cellstudio.models.adapters.ultralytics_adapter
import cellstudio.models.adapters.mmdet_adapter
import cellstudio.pipeline.transforms.loading
import cellstudio.pipeline.transforms.formatting
import cellstudio.pipeline.transforms.visual_aug
import cellstudio.metrics
import cellstudio.plotting  # noqa: F401

def test_1epoch(config_path, work_dir):
    """Run 1 epoch train + val and verify validation happened."""
    print(f"\n{'='*60}")
    print(f"  Testing: {os.path.basename(config_path)}")
    print(f"{'='*60}")
    
    cfg = Config.fromfile(config_path)
    
    # Override to 1 epoch + val every epoch
    cfg.runner = {'type': 'EpochBasedRunner', 'max_epochs': 1, 'val_interval': 1}
    cfg.work_dir = work_dir
    os.makedirs(work_dir, exist_ok=True)
    
    task_cfg = dict(cfg.get('task', {}))
    task_type = task_cfg.pop('type')
    task = TASK_REGISTRY.build({'type': task_type, 'cfg': cfg})
    
    task.execute(mode='train')
    
    # Verify validation ran
    import json
    scalars_path = os.path.join(work_dir, 'scalars.json')
    if os.path.exists(scalars_path):
        lines = open(scalars_path).readlines()
        train_lines = [l for l in lines if '"mode": "train"' in l]
        val_lines = [l for l in lines if '"mode": "val"' in l]
        print(f"\n  Results: {len(train_lines)} train + {len(val_lines)} val entries")
        if val_lines:
            last_val = json.loads(val_lines[-1])
            print(f"  Last val metrics: {last_val}")
            print(f"  >>> PASS <<<")
            return True
        else:
            print(f"  >>> FAIL: No validation entries! <<<")
            return False
    else:
        print(f"  >>> FAIL: No scalars.json <<<")
        return False

if __name__ == '__main__':
    config = 'configs/detect/yolo_v8m_det_mido_tile.yaml'
    result = test_1epoch(config, 'work_dirs/_test_val')
    
    if result:
        # Clean up test dir
        shutil.rmtree('work_dirs/_test_val', ignore_errors=True)
        print("\n=== ALL TESTS PASSED ===")
    else:
        print("\n=== TEST FAILED ===")
        sys.exit(1)
