from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch

@dataclass
class InstanceData:
    """Precision struct for an image's medical instances (cells/tumors)."""
    bboxes: Optional[torch.Tensor] = None
    masks: Optional[Any] = None # Support Tensor or BitmapMask structural types
    labels: Optional[torch.Tensor] = None

@dataclass
class DataSample:
    """
    The Universal Data Capsule.
    Serves as the unbreakable contract containing GT and spatial context for one logical test item.
    """
    img_path: str = ""
    img_shape: tuple = (0, 0)
    ori_shape: tuple = (0, 0)
    
    gt_instances: Optional[InstanceData] = None
    pred_instances: Optional[InstanceData] = None
    
    metainfo: Optional[Dict[str, Any]] = None

@dataclass
class CellStudioInferResult:
    """
    The Supreme Inference Type.
    Adapters map whatever obscure legacy format they use (Ultralytics Tuple / Cellpose dict) 
    into this rigid interface, terminating ambiguity dead in its tracks.
    """
    bboxes: Optional[torch.Tensor] = None
    masks: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
