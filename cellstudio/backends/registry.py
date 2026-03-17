from omegaconf import DictConfig
from pathstudio.backends.ultralytics_yolo.adapter import YoloAdapter

from pathstudio.backends.pytorch_timm.adapter import TimmAdapter

class BackendAdapterRegistry:
    """Registry for dynamically routing tasks to the correct Backend Adapter."""
    
    _adapters = {
        "yolo": YoloAdapter,
        "timm": TimmAdapter,
        # Future: "unet": UnetAdapter
    }
    
    @classmethod
    def get(cls, backend_name: str, config: DictConfig):
        backend_name = backend_name.lower()
        if backend_name not in cls._adapters:
            raise NotImplementedError(f"Backend '{backend_name}' is not supported yet.")
        adapter_cls = cls._adapters[backend_name]
        return adapter_cls(config)
