from omegaconf import DictConfig
from pathstudio.backends.registry import BackendAdapterRegistry


class Predictor:
    """Unified Predictor interface for batch inference across adapters."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.task_type = config.task
        self.backend_name = config.model.backend
        self.weights = config.model.pretrained_weights
        
        # Override source/output if set dynamically by CLI tools
        self.source = getattr(config, "source", "")
        self.output = getattr(config, "output", "./results")
    
    @classmethod
    def from_config(cls, config: DictConfig):
        return cls(config)
        
    def predict_and_save(self):
        print(f"[{self.task_type.upper()}] Predicting from: {self.source} using {self.backend_name}")
        
        adapter = BackendAdapterRegistry.get(self.backend_name, self.config)
        results = adapter.predict(source=self.source, project=self.output)
        
        print(f"Predictions logically completed. Handled by backend saving utility.")
        return results

