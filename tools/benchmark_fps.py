import argparse
import sys
import os
import torch
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cellstudio.engine.config.config import Config
from cellstudio.models.builder import MODEL_REGISTRY
import cellstudio.models.adapters.timm_adapter
import cellstudio.models.adapters.ultralytics_adapter

def parse_args():
    parser = argparse.ArgumentParser(description='Compute Inference FPS for a Zenith Model Config')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--shape', type=int, nargs='+', default=[16, 3, 224, 224], help='input batch shape (def: 16)')
    parser.add_argument('--warmup', type=int, default=50, help='warmup iterations')
    parser.add_argument('--iters', type=int, default=200, help='measurement iterations')
    parser.add_argument('--fp16', action='store_true', help='Use half precision (AMP)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    model_cfg = cfg.get('model')
    model = MODEL_REGISTRY.build(model_cfg)
    
    if hasattr(model, 'model'):
        core_model = model.model
    else:
        core_model = model
        
    core_model.eval()
    if torch.cuda.is_available():
        core_model = core_model.cuda()
    
    device = next(core_model.parameters()).device
    dummy_input = torch.randn(*args.shape).to(device)
    
    if args.fp16:
        core_model = core_model.half()
        dummy_input = dummy_input.half()
        print("Enabled FP16 Engine.")

    print(f"Warming up for {args.warmup} iterations...")
    with torch.no_grad():
        for _ in range(args.warmup):
            core_model(dummy_input)

    print(f"Measuring over {args.iters} iterations...")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    timings = []
    with torch.no_grad():
        for _ in range(args.iters):
            starter.record()
            core_model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))

    avg_time_ms = sum(timings) / len(timings)
    batch_size = args.shape[0]
    fps = (1000.0 / avg_time_ms) * batch_size
    
    print(f"=========================================")
    print(f"Model: {model_cfg.get('architecture', model_cfg.get('yaml_model', 'Unknown'))}")
    print(f"Batch Size: {batch_size}")
    print(f"Avg Batch Time: {avg_time_ms:.2f} ms")
    print(f"Throughput (FPS): {fps:.2f} img/s")
    print(f"=========================================")

if __name__ == '__main__':
    main()
