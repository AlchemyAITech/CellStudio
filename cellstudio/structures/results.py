"""Standardized data structures for CellStudio inference results.

These dataclasses serve as the universal contract between model
adapters and the evaluation / visualization subsystems.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class InstanceData:
    """Per-image instance-level annotations or predictions.

    Attributes:
        bboxes: ``(N, 4)`` tensor of bounding boxes in
            ``[x1, y1, x2, y2]`` format.
        masks: Binary instance masks.  May be a ``Tensor`` of shape
            ``(N, H, W)`` or a ``BitmapMask``-style object.
        labels: ``(N,)`` tensor of integer class labels.
    """

    bboxes: Optional[torch.Tensor] = None
    masks: Optional[Any] = None
    labels: Optional[torch.Tensor] = None


@dataclass
class DataSample:
    """Universal data capsule for a single image sample.

    Encapsulates image metadata, ground-truth annotations, and model
    predictions in a single object that flows through the pipeline.

    Attributes:
        img_path: Absolute path to the source image file.
        img_shape: ``(H, W)`` after any resize / crop transforms.
        ori_shape: ``(H, W)`` of the original image on disk.
        gt_instances: Ground-truth annotations.
        pred_instances: Model predictions.
        metainfo: Arbitrary key-value metadata (e.g. scanner type,
            stain protocol).
    """

    img_path: str = ""
    img_shape: tuple = (0, 0)
    ori_shape: tuple = (0, 0)
    gt_instances: Optional[InstanceData] = None
    pred_instances: Optional[InstanceData] = None
    metainfo: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class CellStudioInferResult:
    """Standardized inference output from any model adapter.

    Adapters must map their framework-specific outputs (Ultralytics
    tuples, Cellpose dicts, etc.) into this rigid interface so that
    downstream evaluation and visualization code can operate without
    any backend-specific branching.

    Attributes:
        bboxes: ``(N, 4)`` detected bounding boxes.
        masks: ``(N, H, W)`` instance segmentation masks.
        labels: ``(N,)`` predicted class labels.
        scores: ``(N,)`` confidence scores.
    """

    bboxes: Optional[torch.Tensor] = None
    masks: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
