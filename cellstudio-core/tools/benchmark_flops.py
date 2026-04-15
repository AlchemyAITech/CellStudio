import argparse
import sys
import os
import torch
from thop import profile, clever_format

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cellstudio.engine.config.config import Config
from cellstudio.models.builder import MODEL_REGISTRY
import cellstudio.models.adapters.timm_adapter
import cellstudio.models.adapters.ultralytics_adapter  # noqa: F401

def parse_args():
    parser = argparse.ArgumentParser(description='Compute FLOPs and Params for a Zenith Model Config')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--shape', type=int, nargs='+', default=[1, 3, 224, 224], help='input shape')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    # 1. Build Model
    model_cfg = cfg.get('model')
    model = MODEL_REGISTRY.build(model_cfg)
    
    # Expose the bare PyTorch model (if it's an adapter)
    if hasattr(model, 'model'):
        core_model = model.model
    else:
        core_model = model
        
    core_model.eval()
    if torch.cuda.is_available():
        core_model = core_model.cuda()
    
    # 2. Dummy Input
    device = next(core_model.parameters()).device
    dummy_input = torch.randn(*args.shape).to(device)
    
    # 3. Profiling
    try:
        flops, params = profile(core_model, inputs=(dummy_input, ), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        print(f"=========================================")
        print(f"Model: {model_cfg.get('architecture', model_cfg.get('yaml_model', 'Unknown'))}")
        print(f"Input Shape: {args.shape}")
        print(f"FLOPs: {flops_str}")
        print(f"Params: {params_str}")
        print(f"=========================================")
    except Exception as e:
        print(f"Failed to profile: {e}")

if __name__ == '__main__':
    main()
