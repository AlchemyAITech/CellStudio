import argparse
from pathstudio.configs.schema import load_config
from pathstudio.engine.exporter import Exporter

def parse_args():
    parser = argparse.ArgumentParser(description="Export a model for deployment")
    parser.add_argument("--config", type=str, required=True, help="Config file matching the training configuration")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights file")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "tensorrt", "torchscript"], help="Export format target")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    config.model.pretrained_weights = args.weights
    config.model.export_format = args.format
    
    exporter = Exporter.from_config(config)
    exporter.export()

if __name__ == "__main__":
    main()
