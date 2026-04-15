import os

import torch
from omegaconf import DictConfig

from cellstudio.backends.registry import BackendAdapterRegistry


class Tester:
    """Unified Tester interface for evaluating model metrics across different backends."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.task_type = config.task
        self.backend_name = config.backend
        self.weights = config.model.get("pretrained_weights", None)
        
        # GPU Support Detection
        self.device = config.env.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Tester] Initializing with device: {self.device.upper()}")
        if self.device == "cuda":
            try:
                print(f"[Tester] CUDA Device Name: {torch.cuda.get_device_name(0)}")
            except AssertionError:
                print("[Tester] WARNING: Torch requested CUDA but compiled without it. Falling back to CPU.")
                self.device = "cpu"
    
    @classmethod
    def from_config(cls, config: DictConfig):
        return cls(config)
        
    def evaluate(self, data_path: str = None):
        print(f"[{self.task_type.upper()}] Evaluation initiated on {self.backend_name} ({self.device}) with weights: {self.weights}")
        adapter = BackendAdapterRegistry.get(self.backend_name, self.config, device=self.device)
        
        if not data_path:
            if hasattr(self.config.data, "val_path"):
                data_path = self.config.data.val_path
            elif hasattr(self.config.data, "data_dir"):
                json_path = os.path.join(self.config.data.data_dir, "dataset.json")
                yaml_path = os.path.join(self.config.data.data_dir, "data.yaml")
                if os.path.exists(json_path):
                    data_path = json_path
                elif os.path.exists(yaml_path):
                    data_path = yaml_path
                else:
                    data_path = self.config.data.data_dir
            else:
                raise KeyError("Tester requires data_path argument or config.data.val_path/data_dir")
            
        metrics = adapter.evaluate(data_path=data_path)
        
        print(f"[{self.task_type.upper()}] Evaluation Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f" - {k}: {v:.4f}")
            else:
                pass # Skip printing raw prediction arrays
        return metrics

