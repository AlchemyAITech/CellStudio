import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class UNetPlugin(nn.Module):
    """
    Modular UNet Wrapper for CellStudio using `segmentation-models-pytorch`.
    """
    def __init__(
        self,
        architecture: str = 'unet',
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        **kwargs
    ):
        """
        Args:
            architecture: Optional['unet', 'unetplusplus', 'manet', 'linknet', 'fpn', 'pspnet', 'deeplabv3', 'deeplabv3plus', 'pan']
            encoder_name: Name of the classification model that will be used as an encoder (e.g. 'resnet34', 'efficientnet-b0')
            encoder_weights: Pre-trained weights (e.g. 'imagenet'). `None` for random initialization.
            in_channels: A number of input channels for the model.
            classes: A number of classes for output (output shape - `(batch, classes, h, w)`).
            **kwargs: Extra architectural args
        """
        super().__init__()
        self.architecture = architecture.lower()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        
        self.model = self._build_model(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **kwargs
        )
        
    def _build_model(self, encoder_name, encoder_weights, in_channels, classes, **kwargs):
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation-models-pytorch library is required. "
                "`pip install segmentation-models-pytorch`"
            )

        arch_map = {
            'unet': smp.Unet,
            'unetplusplus': smp.UnetPlusPlus,
            'unet++': smp.UnetPlusPlus,
            'manet': smp.MAnet,
            'linknet': smp.Linknet,
            'fpn': smp.FPN,
            'pspnet': smp.PSPNet,
            'deeplabv3': smp.DeepLabV3,
            'deeplabv3plus': smp.DeepLabV3Plus,
            'pan': smp.PAN
        }
        
        if self.architecture not in arch_map:
            raise ValueError(f"Unsupported architecture '{self.architecture}'. Choose from {list(arch_map.keys())}")
            
        print(f"[UNet Plugin] Building {self.architecture} with encoder '{encoder_name}' (weights={encoder_weights})...")
        ModelClass = arch_map[self.architecture]
        return ModelClass(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **kwargs
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, C, H, W).
        Returns:
            Output mask tensor of shape (B, classes, H, W).
        """
        return self.model(x)
