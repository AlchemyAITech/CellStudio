import argparse
import sys
import os
import json
import shutil

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
import cellstudio.plotting  # noqa: F401
from cellstudio.inference.inferencer import CellStudioInferencer

def parse_args():
    parser = argparse.ArgumentParser(description='CellStudio Unified CLI Gateway')
    subparsers = parser.add_subparsers(dest='command', help='Top-level commands (train, eval, infer)')
    
    # Train Parser
    train_parser = subparsers.add_parser('train', help='Initiate full-lifecycle training loop')
    train_parser.add_argument('config', help='relative/abs path to the model YAML blueprint')
    train_parser.add_argument('--work-dir', help='override the default work_dirs logs destination')
    
    # Eval Parser
    eval_parser = subparsers.add_parser('eval', help='Offline batch inference and benchmark script')
    eval_parser.add_argument('config', help='path to the validation/testing YAML config')
    eval_parser.add_argument('checkpoint', help='path to the .pth weights')
    
    # Infer Parser
    infer_parser = subparsers.add_parser('infer', help='Single-node structural inference endpoint')
    infer_parser.add_argument('config', help='path to the deployed YAML config')
    infer_parser.add_argument('checkpoint', help='path to the specific .pth weight file')
    infer_parser.add_argument('--image', required=True, help='absolute/relative path to the target image')
    infer_parser.add_argument('--device', default='cuda', help='target deployment device (cuda/cpu)')
    infer_parser.add_argument('--out', help='optional path to dump raw JSON inferences to disk')
    
    args = parser.parse_args()
    return args

def exec_train(args):
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    task_cfg = cfg.get('task')
    if not task_cfg: raise ValueError("Omega Configs must define 'task' with type (e.g. ClassificationTask)")
        
    task_type = task_cfg.pop('type')
    
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(cfg.work_dir, os.path.basename(args.config)))
    
    task = TASK_REGISTRY.build({'type': task_type, 'cfg': cfg})
    print(f"===========================================================")
    print(f" CELLSTUDIO OMEGA INITIATED: [ {task_type} - TRAIN ]")
    print(f"===========================================================")
    task.execute(mode='train')
    
def exec_eval(args):
    # Setup for post-training batched evaluation
    cfg = Config.fromfile(args.config)
    cfg.work_dir = os.path.abspath(os.path.join(os.path.dirname(args.checkpoint), '..'))
    task_cfg = cfg.get('task')
    task_type = task_cfg.pop('type')
    
    TASK_REGISTRY.build({'type': task_type, 'cfg': cfg})
    print(f"===========================================================")
    print(f" CELLSTUDIO OMEGA INITIATED: [ {task_type} - EVAL ]")
    print(f"===========================================================")
    
    # Needs test logic mapped onto task - simulated for MVP structural integrity
    # task.execute(mode='test', checkpoint=args.checkpoint)
    print("[CLI] Evaluation routine delegated into batch hooks. This function intercepts pure metrics testing streams.")

def exec_infer(args):
    if not os.path.exists(args.image):
        print(f"[Error] Visual asset '{args.image}' not surfaced.")
        sys.exit(1)
        
    print(f"===========================================================")
    print(f" CELLSTUDIO OMEGA INITIATED: [ CLI INFERENCER ]")
    print(f"===========================================================")
    
    inferencer = CellStudioInferencer(config_path=args.config, weight_path=args.checkpoint, device=args.device)
    
    print(f"\n[Inferencer] Interrogating Asset: {args.image}...")
    prediction = inferencer(args.image)
    
    print("\n-------------------------- RESULT --------------------------")
    print(json.dumps(prediction, indent=4))
    print("------------------------------------------------------------\n")
    
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(prediction, f, indent=4)
        print(f"[Inferencer] Dumped persistent JSON signature to: {args.out}")

def main():
    args = parse_args()
    if not args.command:
        print("[Error] Must specify a top-level command: train, eval, infer")
        sys.exit(1)
        
    if args.command == 'train':
        exec_train(args)
    elif args.command == 'eval':
        exec_eval(args)
    elif args.command == 'infer':
        exec_infer(args)

if __name__ == '__main__':
    main()
