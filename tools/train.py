import argparse
from pathstudio.configs.schema import load_config
from pathstudio.engine.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PathStudio model")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration file (.yaml)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (e.g., cuda, cpu)")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Optional: Merge CLI device override into config
    config.env.device = args.device

    # Initialize unified trainer
    trainer = Trainer.from_config(config)
    
    # Start training workflow (dataset -> adapter -> training loop)
    trainer.train()

if __name__ == "__main__":
    main()
