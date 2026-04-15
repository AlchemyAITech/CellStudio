"""Anti-corruption adapter interface for third-party model backends.

All backend adapters (YOLO, UNet, Timm, Cellpose, etc.) **must**
subclass :class:`BaseBackendAdapter` to ensure the engine layer
remains completely agnostic to the underlying framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseBackendAdapter(ABC):
    """Abstract adapter that insulates CellStudio from 3rd-party APIs.

    Each concrete adapter wraps one external framework (Ultralytics,
    timm, Cellpose, etc.) and exposes a uniform train/evaluate/predict/
    export interface to the engine.

    Args:
        config: The full resolved configuration dictionary.
        device: Target compute device (e.g. ``'cuda'``, ``'cpu'``).

    Attributes:
        model: The underlying 3rd-party model instance, built by
            :meth:`_build_model`.
        hook_manager: A :class:`HookManager` for adapter-level hooks.
    """

    def __init__(self, config: Dict, device: Optional[str] = None) -> None:
        self.config = config
        self.device = device
        self.model = self._build_model()

        from cellstudio.engine.hooks import HookManager
        self.hook_manager = HookManager()

    def register_hook(self, hook: Any) -> None:
        """Register a hook to the adapter's hook manager.

        Args:
            hook: A hook instance to be invoked during training events.
        """
        self.hook_manager.add_hook(hook)

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by every adapter
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_model(self) -> Any:
        """Initialize and return the underlying 3rd-party model.

        Returns:
            The constructed model object.
        """

    @abstractmethod
    def train(self, data_path: str, **kwargs) -> Dict:
        """Execute the training loop.

        Args:
            data_path: Path to the training dataset or data config.
            **kwargs: Backend-specific training options.

        Returns:
            Dictionary containing training results and output paths
            (e.g. ``{'save_dir': '...', 'best_mAP': 0.85}``).
        """

    @abstractmethod
    def evaluate(self, data_path: str, **kwargs) -> Dict:
        """Execute the evaluation loop.

        Args:
            data_path: Path to the evaluation dataset or data config.
            **kwargs: Backend-specific evaluation options.

        Returns:
            Dictionary of computed metrics.
        """

    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> Any:
        """Run inference on inputs and return standardized results.

        Args:
            inputs: Input data (image path, tensor, or batch).
            **kwargs: Backend-specific inference options.

        Returns:
            Prediction results in CellStudio's standard format.
        """

    @abstractmethod
    def export(self, export_format: str, save_path: str, **kwargs) -> str:
        """Export the model to an interchange format.

        Args:
            export_format: Target format (``'onnx'``, ``'torchscript'``,
                ``'tensorrt'``).
            save_path: File path for the exported artifact.
            **kwargs: Format-specific export options.

        Returns:
            Path to the exported model file.
        """
