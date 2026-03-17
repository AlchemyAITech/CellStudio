import os
from omegaconf import DictConfig
from pathstudio.backends.registry import BackendAdapterRegistry

class Tester:
    """Unified Tester interface for evaluating model metrics across different backends."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.task_type = config.task
        self.backend_name = config.model.backend
        self.weights = config.model.pretrained_weights
    
    @classmethod
    def from_config(cls, config: DictConfig):
        return cls(config)
        
    def evaluate(self):
        print(f"[{self.task_type.upper()}] Evaluation initiated on {self.backend_name} with weights: {self.weights}")
        adapter = BackendAdapterRegistry.get(self.backend_name, self.config)
        
        json_path = os.path.join(self.config.data.data_dir, "dataset.json")
        yaml_path = os.path.join(self.config.data.data_dir, "data.yaml")
        
        if os.path.exists(json_path):
            data_path = json_path
        elif os.path.exists(yaml_path):
            data_path = yaml_path
        else:
            data_path = self.config.data.data_dir
            
        metrics = adapter.evaluate(data_path=data_path)
        
        print(f"[{self.task_type.upper()}] Evaluation Results:")
        for k, v in metrics.items():
            print(f" - {k}: {v:.4f}")
        return metrics

