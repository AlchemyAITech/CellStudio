import argparse
import sys
import os

# Link CellStudio Python path entirely explicitly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellstudio.engine.config.config import Config
from cellstudio.tasks.registry import TASK_REGISTRY

# Registry injection calls directly replacing verbose monolithic imports
import cellstudio.tasks.object_detection
import cellstudio.tasks.classification
import cellstudio.models.adapters.ultralytics_adapter
import cellstudio.models.adapters.timm_adapter
import cellstudio.pipeline.transforms.loading
import cellstudio.pipeline.transforms.formatting
import cellstudio.pipeline.transforms.visual_aug
import cellstudio.metrics
import cellstudio.plotting

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

    task_cfg = cfg.get('task')
    if not task_cfg:
        raise ValueError("Omega Configs must unconditionally define 'task' with type (e.g. ObjectDetectionTask)")
        
    task_type = task_cfg.pop('type')
    
    task = TASK_REGISTRY.build({'type': task_type, 'cfg': cfg})
    print(f"===========================================================")
    print(f" ZENITH ARCHITECTURE OMEGA INITIATED: [ {task_type} ]")
    print(f"===========================================================")
    task.execute(mode='train')

if __name__ == '__main__':
    main()
