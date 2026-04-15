import argparse
import sys
import os
import json

# Link CellStudio Python path entirely explicitly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellstudio.inference.inferencer import CellStudioInferencer

# Injection calls needed to populate model registries inside the CLI scope
import cellstudio.models.adapters.ultralytics_adapter
import cellstudio.models.adapters.timm_adapter  # noqa: F401

def parse_args():
    parser = argparse.ArgumentParser(description='CellStudio CLI Inferencer Tool')
    parser.add_argument('config', help='path to the validation/testing YAML config')
    parser.add_argument('checkpoint', help='path to the specific .pth weight file')
    parser.add_argument('--image', required=True, help='absolute/relative path to the target image')
    parser.add_argument('--device', default='cuda', help='target deployment device (cuda/cpu)')
    parser.add_argument('--out', help='optional path to dump raw JSON inferences to disk')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    if not os.path.exists(args.image):
        print(f"[Error] Visual asset '{args.image}' not surfaced.")
        sys.exit(1)
        
    print("===========================================================")
    print(" ZENITH ARCHITECTURE OMEGA INITIATED: [ CLI Inferencer ]")
    print("===========================================================")
    
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

if __name__ == '__main__':
    main()
