from omegaconf import DictConfig
from pathstudio.backends.registry import BackendAdapterRegistry

class Exporter:
    """Unified module entry to export PyTorch or other backend weights to ONNX/TRT."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.task_type = config.task
        self.backend_name = config.model.backend
        self.weights = config.model.pretrained_weights
        self.export_format = config.model.export_format
    
    @classmethod
    def from_config(cls, config: DictConfig):
        return cls(config)
        
    def export(self):
        print(f"[{self.task_type.upper()}] Exporting {self.weights} to {self.export_format.upper()} format using {self.backend_name}.")
        
        adapter = BackendAdapterRegistry.get(self.backend_name, self.config)
        saved_path = adapter.export(export_format=self.export_format)
        
        print(f"Export completed successfully. Saved to: {saved_path}")
        return saved_path

