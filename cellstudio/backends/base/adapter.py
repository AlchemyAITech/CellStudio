from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseBackendAdapter(ABC):
    """
    The Anti-corruption Layer for all third-party models.
    All backend adapters (YOLO, UNet, Timm, etc.) MUST implement this interface
    to ensure the engine layer remains completely agnostic to the underlying framework.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = self._build_model()
        from pathstudio.engine.hooks import HookManager
        self.hook_manager = HookManager()
        
    def register_hook(self, hook: Any):
        """Register a hook to the adapter's hook manager."""
        self.hook_manager.add_hook(hook)
        
    @abstractmethod
    def _build_model(self) -> Any:
        """Initialize the underlying 3rd-party model."""
        pass
        
    @abstractmethod
    def train(self, data_module: Any, **kwargs) -> Dict:
        """Execute the training loop. Must return standard metrics/status."""
        pass

    @abstractmethod
    def evaluate(self, data_module: Any, **kwargs) -> Dict:
        """Execute the evaluation loop. Must return standard metrics."""
        pass
        
    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> Any:
        """Run inference and return standard format."""
        pass
        
    @abstractmethod
    def export(self, export_format: str, save_path: str, **kwargs) -> str:
        """Export the model to ONNX, TensorRT, TorchScript."""
        pass
