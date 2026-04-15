import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np
from .base import BaseModelAdapter
from ..builder import MODEL_REGISTRY
from ...structures.results import DataSample, CellStudioInferResult, InstanceData

@MODEL_REGISTRY.register('UltralyticsSegAdapter')
class UltralyticsSegAdapter(BaseModelAdapter):
    """
    Adapter for Ultralytics YOLOv8-Seg models.
    Supports Box Loss, Cls Loss, DFL Loss, and Mask Loss.
    """
    def __init__(self, yaml_model: str = "yolov8m-seg.yaml", pretrained: bool = False, num_classes: int = 2):
        super().__init__()
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg

        _yolo = YOLO(yaml_model)
        self.model = _yolo.model
        
        self.model.args = get_cfg()
        self.model.args.model = yaml_model
        self.model.args.nc = num_classes
        self.model.args.overlap_mask = True # Required for seg
        
        # Unfreeze all layers
        for p in self.model.parameters():
            p.requires_grad = True
            
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.loss_fn = self.model.init_criterion()
        
        if torch.cuda.is_available():
            self.loss_fn.device = torch.device('cuda')

    def _format_yolo_seg_targets(self, data_samples: List[DataSample], img_shape: tuple) -> dict:
        device = next(self.model.parameters()).device
        batch_indices = []
        class_labels = []
        norm_bboxes = []
        masks_list = []
        
        H, W = img_shape
        
        for i, sample in enumerate(data_samples):
            boxes, labels, masks = None, None, None
            if isinstance(sample, dict):
                boxes = sample.get('gt_bboxes')
                labels = sample.get('gt_labels')
                masks = sample.get('gt_masks')
            elif sample.gt_instances is not None:
                boxes = sample.gt_instances.bboxes
                labels = sample.gt_instances.labels
                masks = sample.gt_instances.masks
                
            if boxes is not None and len(boxes) > 0:
                batch_indices.extend([i] * len(boxes))
                class_labels.extend(labels.tolist() if getattr(labels, 'tolist', None) else labels if isinstance(labels, list) else [0]*len(boxes))
                for box in boxes:
                    if hasattr(box, 'tolist'): box = box.tolist()
                    xmin, ymin, xmax, ymax = box
                    x_c = (xmin + xmax) / 2.0 / W
                    y_c = (ymin + ymax) / 2.0 / H
                    w = (xmax - xmin) / W
                    h = (ymax - ymin) / H
                    norm_bboxes.append([x_c, y_c, w, h])
                    
                if getattr(self.model, 'overlap', False):
                    import torch.nn.functional as F
                    if masks is not None:
                        if isinstance(masks, np.ndarray): masks = torch.from_numpy(masks)
                        if masks.ndim == 2:
                            masks = masks.unsqueeze(0)
                    if masks is not None and masks.shape[0] > 0:
                        # masks: [N, H, W] of bool -> [N, 1, H, W] float
                        m_float = masks.unsqueeze(1).float()
                        # Nearest interpolation to preserve small masks
                        m_down = F.interpolate(m_float, size=(int(H // 4), int(W // 4)), mode='nearest').squeeze(1).to(device)
                        # YOLOv8 target_masks must be [BS, H/4, W/4] with instance IDs (1 to N) because overlap_mask=True
                        ids = torch.arange(1, m_down.shape[0] + 1, device=device, dtype=torch.float32).view(-1, 1, 1)
                        m_down_id = (m_down * ids).max(dim=0)[0]  # [H/4, W/4]
                        masks_list.append(m_down_id.unsqueeze(0))
                    else:
                        masks_list.append(torch.zeros((1, H // 4, W // 4), device=device, dtype=torch.float32))
                else:
                    masks_list.append(torch.zeros((1, H // 4, W // 4), device=device, dtype=torch.float32))
                        
        res = {
            'batch_idx': torch.tensor(batch_indices, dtype=torch.long, device=device),
            'cls': torch.tensor(class_labels, dtype=torch.float32, device=device).unsqueeze(1),
            'bboxes': torch.tensor(norm_bboxes, dtype=torch.float32, device=device).reshape(-1, 4)
        }
        if len(masks_list) > 0:
            res['masks'] = torch.cat(masks_list, dim=0).to(device)
        else:
            res['masks'] = torch.zeros((0, img_shape[0] // 4, img_shape[1] // 4), dtype=torch.float32, device=device)
        return res

    def forward_train(self, imgs: torch.Tensor, data_samples: Optional[List[DataSample]]) -> Dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        imgs = imgs.to(device)
        
        preds = self.model(imgs)
        
        if isinstance(preds, dict) and 'one2many' in preds:
            preds = preds['one2many']
            
        batch = self._format_yolo_seg_targets(data_samples, imgs.shape[2:])
        
        if len(batch['bboxes']) > 0:
            loss_components, loss_items = self.loss_fn(preds, batch)
            bs = imgs.shape[0]
            return {
                'loss': loss_components.sum() / bs,
                'box_loss': loss_items[0],
                'cls_loss': loss_items[1],
                'dfl_loss': loss_items[2],
                'seg_loss': loss_items[3]
            }
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {'loss': loss}

    def forward_test(self, imgs: torch.Tensor, data_samples: Optional[List[DataSample]] = None) -> List[CellStudioInferResult]:
        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.ops import process_mask, scale_boxes, scale_masks
        
        device = next(self.model.parameters()).device
        imgs = imgs.to(device)
        
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            preds = self.model(imgs)
            
        if was_training:
            self.model.train()
        
        if isinstance(preds, (tuple, list)):
            pred_tensors = preds[0]
            if len(preds) > 1 and isinstance(preds[1], (tuple, list)) and len(preds[1]) > 0:
                proto = preds[1][-1]
            elif len(preds) > 1 and isinstance(preds[1], dict):
                proto = preds[1].get('proto', next(reversed(preds[1].values())) if len(preds[1]) > 0 else None)
            else:
                proto = None
        else:
            return []
            
        true_nc = getattr(self.model.model[-1], 'nc', self.model.args.nc)
        # NMS - Tuned conf_thres to 0.05: Allows early-epoch metrics visibility while strictly avoiding 0.001 OOMs
        out = non_max_suppression(pred_tensors, conf_thres=0.05, iou_thres=0.6, nc=true_nc)
        
        results = []
        for i, det in enumerate(out):
            if det is not None and len(det):
                bboxes = det[:, :4].cpu()
                scores = det[:, 4].cpu()
                labels = det[:, 5].long().cpu()
                
                # Masks processing
                if proto is not None and det.shape[1] > 6:
                    masks_in = det[:, 6:]
                    masks = process_mask(proto[i], masks_in, bboxes, imgs.shape[2:], upsample=True)
                    masks = masks > 0.5
                else:
                    masks = torch.zeros((len(det), imgs.shape[2], imgs.shape[3]), dtype=torch.bool)
            else:
                bboxes = torch.zeros((0, 4))
                scores = torch.zeros((0,))
                labels = torch.zeros((0,), dtype=torch.long)
                masks = torch.zeros((0, imgs.shape[2], imgs.shape[3]), dtype=torch.bool)
                
            res = CellStudioInferResult(
                bboxes=bboxes,
                labels=labels,
                scores=scores,
                masks=masks.cpu()
            )
            results.append(res)
        return results
