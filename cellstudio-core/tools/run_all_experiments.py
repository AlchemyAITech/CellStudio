import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from cellstudio.engine.trainer import Trainer

def run_classification_experiments():
    models = ["resnet50", "efficientnet_b4", "vit_base_patch16_224"]
    data_dir = r"E:\workspace\AlchemyTech\CellStudio\datasets\classfication\MIDOG\splits"
    
    # Needs a dataset.json in the splits dir for pathstudio unified format
    # For MVP purpose, we'll assume the TimmAdapter can take a CSV path if we enhance it later, or we build a JSON converter.
    # Actually, TimmAdapter currently expects a CellDatasetConfig json. 
    # Let's generate it on the fly if needed, or pass it.
    
    print("\n" + "="*50)
    print("🚀 STARTING CLASSIFICATION EXPERIMENTS")
    print("="*50)
    
    for arch in models:
        print(f"\n--- Running Classification: {arch} ---")
        cfg = OmegaConf.create({
            "task": "classification",
            "env": {"device": "cuda"},
            "model": {
                "backend": "timm",
                "architecture": arch,
                "pretrained": True,
                "pretrained_weights": None
            },
            "data": {
                "data_dir": data_dir,
                "batch_size": 16,
                "num_workers": 4
            },
            "training": {
                "epochs": 100,
                "learning_rate": 0.001,
                "save_dir": "runs/classification"
            }
        })
        try:
            trainer = Trainer.from_config(cfg)
            trainer.train()
        except Exception as e:
            print(f"Failed {arch}: {e}")

def run_detection_experiments():
    models = ["yolov8n", "yolov8s", "yolov8m"]
    data_dir = r"E:\workspace\AlchemyTech\CellStudio\datasets\detection\MIDO\splits"
    
    print("\n" + "="*50)
    print("🚀 STARTING DETECTION EXPERIMENTS")
    print("="*50)
    
    for arch in models:
        print(f"\n--- Running Detection: {arch} ---")
        cfg = OmegaConf.create({
            "task": "detection",
            "env": {"device": "cuda"},
            "model": {
                "backend": "yolo",
                "architecture": arch,
                "pretrained": True,
                "pretrained_weights": None
            },
            "data": {
                "data_dir": data_dir, # Should contain train.json etc. which YoloDataFormatter will convert
                "batch_size": 16,
                "num_workers": 4
            },
            "training": {
                "epochs": 100,
                "learning_rate": 0.01,
                "save_dir": "runs/detection"
            }
        })
        try:
            trainer = Trainer.from_config(cfg)
            trainer.train()
        except Exception as e:
            print(f"Failed {arch}: {e}")

def run_segmentation_experiments():
    # Placeholder for segmentation
    # We will need a UnetAdapter and CellposeAdapter in registry
    print("\n" + "="*50)
    print("🚀 STARTING SEGMENTATION EXPERIMENTS")
    print("="*50)
    print("Segmentation adapter integration is queued for implementation.")

if __name__ == "__main__":
    run_classification_experiments()
    run_detection_experiments()
    run_segmentation_experiments()
