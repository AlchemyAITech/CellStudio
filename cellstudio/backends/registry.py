from omegaconf import DictConfig
from cellstudio.backends.ultralytics_yolo.adapter import YoloAdapter
from cellstudio.backends.pytorch_timm.adapter import TimmAdapter
from cellstudio.backends.unet.adapter import UnetAdapter
from cellstudio.backends.cellpose.adapter import CellposeAdapter

class BackendAdapterRegistry:
    """Registry for dynamically routing tasks to the correct Backend Adapter."""
    
    _adapters = {
        "yolo": YoloAdapter,
        "ultralytics": YoloAdapter,
        "timm": TimmAdapter,
        "unet": UnetAdapter,
        "cellpose": CellposeAdapter,
    }
    
    @classmethod
    def get(cls, backend_name: str, config: DictConfig, device: str = None):
        backend_name = backend_name.lower()
        if backend_name not in cls._adapters:
            raise NotImplementedError(f"Backend '{backend_name}' is not supported yet.")
        adapter_cls = cls._adapters[backend_name]
        return adapter_cls(config, device=device)
