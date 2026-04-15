import torch
import torch.nn as nn
from typing import List, Dict, Optional
from .base import BaseModelAdapter
from ..builder import MODEL_REGISTRY
from ...structures.results import DataSample, CellStudioInferResult

@MODEL_REGISTRY.register('MMDetAdapter')
class MMDetAdapter(BaseModelAdapter):
    def __init__(self, config_file: str, pretrained: str = None, num_classes: int = 2):
        super().__init__()
        from mmdet.registry import MODELS
        from mmengine.config import Config
        from mmdet.utils import register_all_modules
        
        register_all_modules(init_default_scope=True)
        cfg = Config.fromfile(config_file)
        
        # Override num_classes in bbox_head if possible
        if 'bbox_head' in cfg.model:
            if isinstance(cfg.model.bbox_head, list):
                for head in cfg.model.bbox_head:
                    head.num_classes = num_classes
            else:
                cfg.model.bbox_head.num_classes = num_classes
        elif 'roi_head' in cfg.model and 'bbox_head' in cfg.model.roi_head:
            cfg.model.roi_head.bbox_head.num_classes = num_classes
            
        self.model = MODELS.build(cfg.model)
        
        # CRITICAL: Neutralize mmdet's data_preprocessor to prevent double normalization.
        # CellStudio's pipeline already normalizes images (mean subtraction, std division).
        # The built-in DetDataPreprocessor would apply a SECOND normalization, destroying
        # the input distribution. Set mean=0, std=1, no BGR conversion = identity transform.
        if hasattr(self.model, 'data_preprocessor'):
            dp = self.model.data_preprocessor
            if hasattr(dp, 'mean'):
                dp.mean.fill_(0.0)
            if hasattr(dp, 'std'):
                dp.std.fill_(1.0)
            if hasattr(dp, '_enable_normalize'):
                dp._enable_normalize = False
            # Disable BGR-to-RGB conversion since pipeline already provides correct channel order
            if hasattr(dp, 'channel_conversion'):
                dp.channel_conversion = False
        
        if pretrained:
            from mmengine.runner import load_checkpoint
            load_checkpoint(self.model, pretrained, map_location='cpu')

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
        for p in self.model.parameters():
            p.requires_grad = True

    def _format_mmdet_data(self, imgs: torch.Tensor, data_samples: Optional[List[DataSample]]) -> dict:
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData
        
        batch_data_samples = []
        device = next(self.model.parameters()).device
        
        for i, img in enumerate(imgs):
            ds = DetDataSample()
            ds.set_metainfo({
                'img_shape': img.shape[1:],
                'ori_shape': img.shape[1:],
                'pad_shape': img.shape[1:],
                'batch_input_shape': imgs.shape[2:],  # DETR needs this for positional encoding
                'scale_factor': (1.0, 1.0)
            })
            
            if data_samples and data_samples[i]:
                sample = data_samples[i]
                boxes, labels, masks = None, None, None
                if isinstance(sample, dict):
                    boxes = sample.get('gt_bboxes')
                    labels = sample.get('gt_labels')
                    masks = sample.get('gt_masks')
                elif sample.gt_instances is not None:
                    boxes = sample.gt_instances.bboxes
                    labels = sample.gt_instances.labels
                    masks = sample.gt_instances.masks
                    
                instances = InstanceData()
                if boxes is not None and len(boxes) > 0:
                    import numpy as np
                    if isinstance(boxes, list) or isinstance(boxes, np.ndarray):
                        instances.bboxes = torch.tensor(boxes, dtype=torch.float32, device=device)
                    else:
                        instances.bboxes = boxes.clone().detach().to(dtype=torch.float32, device=device)
                        
                    if isinstance(labels, list) or isinstance(labels, np.ndarray):
                        instances.labels = torch.tensor(labels, dtype=torch.long, device=device)
                    else:
                        instances.labels = labels.clone().detach().to(dtype=torch.long, device=device)
                        
                    if masks is not None and len(masks) > 0:
                        if isinstance(masks, list) or isinstance(masks, np.ndarray):
                            from mmdet.structures.mask import BitmapMasks
                            if isinstance(masks, list):
                                # Convert List of Polygons to Stacked Binary Masks [N, H, W]
                                import cv2
                                # Determine image dimensions from instances metainfo or fallback
                                h, w = img.shape[1], img.shape[2] 
                                bit_masks = np.zeros((len(masks), h, w), dtype=np.uint8)
                                for idx, m_poly in enumerate(masks):
                                    poly = np.array(m_poly, dtype=np.int32)
                                    cv2.fillPoly(bit_masks[idx], [poly], 1)
                                masks = bit_masks
                            instances.masks = BitmapMasks(masks, masks.shape[-2], masks.shape[-1])
                        else:
                            instances.masks = masks
                else:
                    instances.bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                    instances.labels = torch.zeros((0,), dtype=torch.long, device=device)
                ds.gt_instances = instances
            batch_data_samples.append(ds)
            
        return {'inputs': imgs.to(device), 'data_samples': batch_data_samples}

    def forward_train(self, imgs: torch.Tensor, data_samples: Optional[List[DataSample]]) -> Dict[str, torch.Tensor]:
        batch = self._format_mmdet_data(imgs, data_samples)
        device = next(self.model.parameters()).device
        
        losses = self.model(batch['inputs'], batch['data_samples'], mode='loss')
        
        # CRITICAL: Only sum keys that contain 'loss' in their name.
        # MMDet also returns 'acc' (~95-100) which is a METRIC, not a loss.
        # Summing acc into the total loss produces loss ~99.x and completely
        # breaks gradient updates, preventing the model from learning.
        graph_loss = None
        for key, val in losses.items():
            if 'loss' not in key:
                continue  # Skip acc, precision, recall, etc.
            if isinstance(val, list):
                for v in val:
                    graph_loss = v if graph_loss is None else graph_loss + v
            elif isinstance(val, torch.Tensor):
                graph_loss = val if graph_loss is None else graph_loss + val
                
        if graph_loss is None:
            graph_loss = torch.tensor(0.0, device=device, requires_grad=True)
        losses['loss'] = graph_loss
            
        # Ensure primitive types for logging
        clean_losses = {}
        for k, v in losses.items():
            if isinstance(v, list):
                clean_losses[k] = sum(v)
            else:
                clean_losses[k] = v
        return clean_losses

    def forward_test(self, imgs: torch.Tensor, data_samples: Optional[List[DataSample]] = None) -> List[CellStudioInferResult]:
        batch = self._format_mmdet_data(imgs, data_samples)
        
        with torch.no_grad():
            preds = self.model(batch['inputs'], batch['data_samples'], mode='predict')
            
        results = []
        for pred in preds:
            instances = pred.pred_instances
            res = CellStudioInferResult(
                bboxes=instances.bboxes.cpu(),
                labels=instances.labels.cpu(),
                scores=instances.scores.cpu()
            )
            results.append(res)
            
        return results
