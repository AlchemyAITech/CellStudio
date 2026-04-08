"""Decoupled inference engine for CellStudio models.

Provides :class:`CellStudioInferencer` which loads a trained model
and its validation pipeline, enabling single-image inference from
both CLI tools and API endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from ..engine.config.config import Config
from ..models.builder import MODEL_REGISTRY
from ..pipeline.compose import Compose

logger = logging.getLogger(__name__)


class CellStudioInferencer:
    """End-to-end inference wrapper for trained CellStudio models.

    Loads a model architecture from config, injects pre-trained weights,
    and assembles the validation pipeline so that callers only need to
    provide an image path.

    Args:
        config_path: Path to the YAML configuration file used during
            training.
        weight_path: Path to the saved model checkpoint (``.pth``).
        device: Target device (``'cuda'`` or ``'cpu'``).  Falls back to
            CPU automatically if CUDA is unavailable.

    Example:
        >>> infer = CellStudioInferencer(
        ...     config_path='configs/classify/resnet50_mido.yaml',
        ...     weight_path='runs/best.pth',
        ... )
        >>> result = infer('data/test_image.png')
        >>> print(result['class_id'], result['confidence'])
    """

    def __init__(
        self,
        config_path: str,
        weight_path: str,
        device: str = 'cuda',
    ) -> None:
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.cfg = Config.fromfile(config_path)

        # 1. Build model architecture
        model_cfg = self.cfg.get('model')
        if not model_cfg:
            raise KeyError("Configuration must contain a 'model' section.")

        self.model = MODEL_REGISTRY.build(model_cfg)
        self.model.to(self.device)
        self.model.eval()

        # 2. Load pre-trained weights
        state_dict = torch.load(
            weight_path, map_location=self.device, weights_only=True,
        )
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        self.model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded weights from %s", weight_path)

        # 3. Assemble inference pipeline (mirrors validation transforms)
        val_loader_cfg = self.cfg.get('val_dataloader', {})
        dataset_cfg = val_loader_cfg.get('dataset', {})
        pipeline_cfg = dataset_cfg.get('pipeline', [])

        if not pipeline_cfg:
            raise ValueError(
                "Configuration must define a dataset pipeline under "
                "'val_dataloader.dataset.pipeline'."
            )

        self.pipeline = Compose(pipeline_cfg)

    @torch.no_grad()
    def __call__(self, img_path: str) -> Dict[str, Any]:
        """Run inference on a single image.

        Args:
            img_path: Path to the input image.

        Returns:
            Dictionary of results.  For classification tasks this
            contains ``'class_id'``, ``'confidence'``, and
            ``'probabilities'``.
        """
        data = dict(img_path=str(img_path))
        data = self.pipeline(data)

        inputs = data['imgs'].unsqueeze(0).to(self.device)

        if hasattr(self.model, 'forward_test'):
            outputs = self.model.forward_test(inputs)
        else:
            outputs = self.model(inputs)

        return self._parse_outputs(outputs)

    @staticmethod
    def _parse_outputs(outputs: Any) -> Dict[str, Any]:
        """Convert raw model outputs to a user-friendly dictionary.

        Args:
            outputs: Raw model output (dict or tensor).

        Returns:
            Parsed result dictionary.
        """
        result: Dict[str, Any] = {}

        if isinstance(outputs, dict) and 'probs' in outputs and 'preds' in outputs:
            probs = outputs['probs'][0].cpu().numpy().tolist()
            pred = int(outputs['preds'][0].item())
            result['class_id'] = pred
            result['confidence'] = probs[pred]
            result['probabilities'] = probs

        return result
