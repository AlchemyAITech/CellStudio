import argparse
from pathstudio.configs.schema import load_config
from pathstudio.engine.tester import Tester

def parse_args():
    parser = argparse.ArgumentParser(description="Test a PathStudio model")
    parser.add_argument("--config", type=str, required=True, help="Path to the training/eval configuration file (.yaml)")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--device", type=str, default="cuda", help="Device to evaluate on (e.g., cuda, cpu)")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    config.model.pretrained_weights = args.weights
    config.env.device = args.device

    tester = Tester.from_config(config)
    metrics = tester.evaluate()
    print("Test Metrics:", metrics)

if __name__ == "__main__":
    main()
