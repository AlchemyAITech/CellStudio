import os
import numpy as np
from typing import List, Dict, Any, Union

class CellposePlugin:
    """
    Cellpose Plugin for CellStudio.
    Provides a unified interface for predicting cell and nucleus segmentation masks
    using the Cellpose library.
    """
    def __init__(self, model_type: str = "cyto", device: str = None):
        """
        model_type: Default cellpose models (e.g. 'cyto', 'nuclei', 'cyto2', 'tissuenet').
        device: 'cuda' or 'cpu'. If None, cellpose will try to auto-detect.
        """
        self.model_type = model_type
        # if device is given explicitly, configure it
        use_gpu = False
        if device is not None and "cuda" in device.lower():
            use_gpu = True
            
        self.use_gpu = use_gpu
        self.model = self._load_model()
        
    def _load_model(self):
        try:
            from cellpose import models, core
        except ImportError:
            raise ImportError("Cellpose library is required. `pip install cellpose`")
            
        print(f"[Cellpose Plugin] Loading model '{self.model_type}' (GPU={self.use_gpu})...")
        
        # Determine GPU usage
        if self.use_gpu and not core.use_gpu():
            print("[Cellpose Plugin] Warning: GPU requested but not available. Falling back to CPU.")
            self.use_gpu = False

        model = models.CellposeModel(gpu=self.use_gpu, model_type=self.model_type)
        return model
        
    def predict(self, 
                image_source: Union[str, np.ndarray], 
                channels: List[int] = [0, 0], 
                diameter: float = None,
                **kwargs) -> Dict[str, Any]:
        """
        Predict cell/nucleus masks for the given image.
        
        Args:
            image_source: Path to an image file or a numpy array (H, W, C).
            channels: list of channels. [0, 0] means grayscale. [2, 1] means Green for cyto, Red for nucleus.
            diameter: Average object diameter in pixels. If None, cellpose automatically estimates it.
            **kwargs: Additional parameters passed to `model.eval()`.
            
        Returns:
            Dictionary containing:
                - 'masks': The instance segmentation mask array (same HW as input).
                - 'flows': Flow arrays from cellpose.
                - 'styles': Style array from cellpose.
                - 'diams': Estimated diameters.
        """
        if isinstance(image_source, str):
            import cv2
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image not found at {image_source}")
            image = cv2.imread(image_source)
            if image is None:
                raise ValueError(f"Failed to read image at {image_source}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_source, np.ndarray):
            image = image_source
        else:
            raise ValueError("image_source must be a valid file path or a numpy array.")

        print(f"[Cellpose Plugin] Running inference on image shape {image.shape} with channels={channels}...")

        # cellpose eval method
        masks, flows, styles, diams = self.model.eval(
            image, 
            diameter=diameter, 
            channels=channels, 
            **kwargs
        )

        return {
            "masks": masks,
            "flows": flows,
            "styles": styles,
            "diams": diams
        }
