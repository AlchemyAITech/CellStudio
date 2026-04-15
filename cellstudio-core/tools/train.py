import argparse
import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Link CellStudio Python path entirely explicitly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellstudio.engine.config.config import Config
from cellstudio.tasks.registry import TASK_REGISTRY

# Registry injection calls directly replacing verbose monolithic imports
import cellstudio.tasks.object_detection
import cellstudio.tasks.classification
import cellstudio.tasks.segmentation
import cellstudio.models.adapters.ultralytics_adapter
import cellstudio.models.adapters.timm_adapter
import cellstudio.models.adapters.mmdet_adapter
import cellstudio.pipeline.transforms.loading
import cellstudio.pipeline.transforms.formatting
import cellstudio.pipeline.transforms.visual_aug
import cellstudio.metrics
import cellstudio.plotting  # noqa: F401

def parse_args():
    parser = argparse.ArgumentParser(description='Execute Zenith Architected Train Loop')
    parser.add_argument('config', help='train config file path relative/abs')
    parser.add_argument('--work-dir', help='the dir to save logs and models override')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    task_cfg = dict(cfg.get('task', {}))
    if not task_cfg:
        raise ValueError("Omega Configs must unconditionally define 'task' with type (e.g. ObjectDetectionTask)")
        
    task_type = task_cfg.pop('type', None)
    if not task_type:
        raise ValueError("task.type must be specified")
    
    if os.path.exists(cfg.work_dir) and cfg.work_dir != './':
        import shutil
        shutil.rmtree(cfg.work_dir)
        
    os.makedirs(cfg.work_dir, exist_ok=True)
    import shutil
    shutil.copy(args.config, os.path.join(cfg.work_dir, os.path.basename(args.config)))
    
    task = TASK_REGISTRY.build({'type': task_type, 'cfg': cfg})
    print(f"===========================================================")
    print(f" ZENITH ARCHITECTURE OMEGA INITIATED: [ {task_type} ]")
    print(f"===========================================================")
    task.execute(mode='train')

if __name__ == '__main__':
    main()
