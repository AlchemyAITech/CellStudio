import argparse
from pathstudio.configs.schema import load_config
from pathstudio.engine.predictor import Predictor

def parse_args():
    parser = argparse.ArgumentParser(description="Predict on images using a PathStudio model")
    parser.add_argument("--config", type=str, required=True, help="Configuration file (defining task, pre/post processing)")
    parser.add_argument("--source", type=str, required=True, help="Path to images or a directory of images")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--device", type=str, default="cuda", help="Device to predict on (e.g., cuda, cpu)")
    parser.add_argument("--output", type=str, default="./results", help="Directory or file to save predictions")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Use OmegaConf runtime dictionary attachment for transient CLI variables
    from omegaconf import OmegaConf
    OmegaConf.set_struct(config, False)
    
    config.source = args.source
    config.model.pretrained_weights = args.weights
    config.env.device = args.device
    config.output = args.output
    
    predictor = Predictor.from_config(config)
    predictor.predict_and_save()

if __name__ == "__main__":
    main()
