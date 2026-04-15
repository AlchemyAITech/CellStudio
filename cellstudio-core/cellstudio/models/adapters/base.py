import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
import torch

from ...structures.results import DataSample, CellStudioInferResult

class BaseModelAdapter(nn.Module, ABC):
    """
    The Zenith Component: BaseModelAdapter.
    The ultimate corruption barrier shielding CellStudio from arbitrary model libraries (Ultralytics, Cellpose).
    Any external model must be wrapped in an adapter implementing these standard interfaces.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward_train(self, imgs: torch.Tensor, data_samples: Optional[List[DataSample]]) -> Dict[str, torch.Tensor]:
        """
        Must ingest strongly typed Tensors/DataSamples and return a dictionary 
        containing at least a 'loss' key, representing the aggregated 
        scalar tensor for backward propagation.
        """
        pass

    @abstractmethod
    def forward_test(self, imgs: torch.Tensor, data_samples: Optional[List[DataSample]] = None) -> List[CellStudioInferResult]:
        """
        Must ingest Tensors and return strongly typed structured `CellStudioInferResult` objects.
        No tuples, no custom dictionaries allowed.
        """
        pass

    def forward(self, imgs: torch.Tensor, data_samples: Optional[List[DataSample]] = None, mode: str = 'train'):
        """
        Overrides the standard PyTorch nn.Module forward pass with explicit operational modes.
        """
        if mode == 'train':
            return self.forward_train(imgs, data_samples)
        elif mode == 'test':
            return self.forward_test(imgs, data_samples)
        else:
            raise ValueError(f"Invalid forward mode: {mode}")
