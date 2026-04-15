import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from .base import BaseModelAdapter
from ..builder import MODEL_REGISTRY
from ...structures.results import DataSample, CellStudioInferResult

@MODEL_REGISTRY.register('UltralyticsDetAdapter')
class UltralyticsDetAdapter(BaseModelAdapter):
    """
    Subjugates the Ultralytics YOLO backend into native Zenith architecture.
    Bypasses ultralytics Trainer class to control the gradients directly.
    """
    def __init__(self, yaml_model: str = "yolov8m.yaml", pretrained: bool = False, num_classes: int = 2):
        super().__init__()
        import torch.nn as nn
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg

        _yolo = YOLO(yaml_model)
        self.model = _yolo.model
        
        # Inject hyperparams for v8DetectionLoss to function
        self.model.args = get_cfg()
        self.model.args.model = yaml_model
        self.model.args.nc = num_classes
        
        # CRITICAL: Rebuild detection head if num_classes differs from pretrained.
        # Pretrained COCO models have nc=80; we need nc=num_classes.
        # Without this, the cls output channels are 80 but loss expects 2,
        # causing stagnant training loss and 0.0 mAP.
        detect_head = self.model.model[-1]  # Detect layer
        if hasattr(detect_head, 'nc') and detect_head.nc != num_classes:
            old_nc = detect_head.nc
            detect_head.nc = num_classes
            self.model.nc = num_classes
            
            def _rebuild_cls_branch(cv3_module, nl, num_cls):
                """Replace final conv in each level of a classification branch."""
                for i in range(nl):
                    old_conv = cv3_module[i][-1]
                    if hasattr(old_conv, 'conv'):
                        in_ch = old_conv.conv.in_channels
                        cv3_module[i][-1] = nn.Conv2d(in_ch, num_cls, 1)
                    elif isinstance(old_conv, nn.Conv2d):
                        in_ch = old_conv.in_channels
                        cv3_module[i][-1] = nn.Conv2d(in_ch, num_cls, 1)
            
            # Rebuild cv3 (one2many classification branch)
            _rebuild_cls_branch(detect_head.cv3, detect_head.nl, num_classes)
            
            # YOLO11 end2end models also have one2one_cv3 for NMS-free inference
            if hasattr(detect_head, 'one2one_cv3'):
                _rebuild_cls_branch(detect_head.one2one_cv3, detect_head.nl, num_classes)
            
            # Update the no (number of outputs per anchor) attribute
            detect_head.no = num_classes + detect_head.reg_max * 4
            
        # Unfreeze all layers for end-to-end task finetuning
        for p in self.model.parameters():
            p.requires_grad = True
            
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.loss_fn = self.model.init_criterion()
        
        if torch.cuda.is_available():
            self.loss_fn.device = torch.device('cuda')

    def forward_train(self, imgs: torch.Tensor, data_samples: Optional[List[DataSample]]) -> Dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        imgs = imgs.to(device)
        
        preds = self.model(imgs)
        
        # YOLO11/26 models output {'one2many': {...}, 'one2one': {...}}
        # while YOLOv8 outputs {'boxes': ..., 'scores': ..., 'feats': ...}
        # Extract one2many branch for training loss, as it contains the dense predictions
        if isinstance(preds, dict) and 'one2many' in preds:
            preds = preds['one2many']
        
        # Translates our DataSample structs into YOLO's custom tensor definition
        batch = self._format_yolo_targets(data_samples, imgs.shape[2:])
        batch['batch_idx'] = batch['batch_idx'].to(device)
        batch['cls'] = batch['cls'].to(device)
        batch['bboxes'] = batch['bboxes'].to(device)
        
        if len(batch['bboxes']) > 0:
            loss_components, loss_items = self.loss_fn(preds, batch)
            bs = imgs.shape[0]
            # loss_components = loss * batch_size (ultralytics convention)
            # loss_items = loss.detach() — already per-sample averaged
            return {
                'loss': loss_components.sum() / bs,
                'box_loss': loss_items[0],
                'cls_loss': loss_items[1],
                'dfl_loss': loss_items[2]
            }
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {'loss': loss}

    def forward_test(self, imgs: torch.Tensor, data_samples: Optional[List[DataSample]] = None) -> List[CellStudioInferResult]:
        from ultralytics.utils.nms import non_max_suppression
        device = next(self.model.parameters()).device
        imgs = imgs.to(device)
        
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            preds = self.model(imgs)
        
        raw = None
        
        # Case 1: eval mode returns tuple — first element is the raw prediction tensor
        if isinstance(preds, (list, tuple)):
            raw = preds[0]
            # If first element is still a dict (YOLO11 eval may return this), extract
            if isinstance(raw, dict) and 'one2many' in raw:
                om = raw['one2many']
                if isinstance(om, (list, tuple)) and len(om) >= 2:
                    raw = torch.cat(om[:2], dim=1)
                elif isinstance(om, dict) and 'boxes' in om and 'scores' in om:
                    raw = torch.cat([om['boxes'], om['scores']], dim=1)
        # Case 2: dict with one2many (train mode output)
        elif isinstance(preds, dict) and 'one2many' in preds:
            om = preds['one2many']
            if isinstance(om, (list, tuple)) and len(om) >= 2:
                raw = torch.cat(om[:2], dim=1)
            elif isinstance(om, dict) and 'boxes' in om and 'scores' in om:
                raw = torch.cat([om['boxes'], om['scores']], dim=1)
        # Case 3: direct tensor
        elif isinstance(preds, torch.Tensor):
            raw = preds
        
        if was_training:
            self.model.train()
        
        if raw is None:
            # Last resort: try train mode
            self.model.train()
            with torch.no_grad():
                preds = self.model(imgs)
            if was_training:
                self.model.train()
            else:
                self.model.eval()
            if isinstance(preds, (list, tuple)):
                raw = preds[0]
            elif isinstance(preds, dict) and 'one2many' in preds:
                om = preds['one2many']
                if isinstance(om, (list, tuple)) and len(om) >= 2:
                    raw = torch.cat(om[:2], dim=1)
            elif isinstance(preds, torch.Tensor):
                raw = preds
        
        if raw is None:
            # Cannot extract predictions at all
            batch_size = imgs.shape[0]
            return [CellStudioInferResult(
                bboxes=torch.zeros((0, 4)),
                labels=torch.zeros((0,), dtype=torch.long),
                scores=torch.zeros((0,))
            ) for _ in range(batch_size)]
        
        out = non_max_suppression(raw, conf_thres=0.001, iou_thres=0.6)
        
        results = []
        for i, det in enumerate(out):
            if det is not None and len(det):
                bboxes = det[:, :4].cpu()
                scores = det[:, 4].cpu()
                labels = det[:, 5].long().cpu()
            else:
                bboxes = torch.zeros((0, 4))
                scores = torch.zeros((0,))
                labels = torch.zeros((0,), dtype=torch.long)
                
            res = CellStudioInferResult(
                bboxes=bboxes,
                labels=labels,
                scores=scores
            )
            results.append(res)
        return results

    def _format_yolo_targets(self, data_samples: List[DataSample], img_shape: tuple) -> dict:
        """Converts CellStudio GT into native YOLO sequence formatting."""
        batch_indices = []
        class_labels = []
        norm_bboxes = []
        
        H, W = img_shape
        
        for i, sample in enumerate(data_samples):
            boxes, labels = None, None
            if isinstance(sample, dict):
                boxes = sample.get('gt_bboxes')
                labels = sample.get('gt_labels')
            elif sample.gt_instances is not None:
                boxes = sample.gt_instances.bboxes
                labels = sample.gt_instances.labels
                
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
                    
        return {
            'batch_idx': torch.tensor(batch_indices, dtype=torch.long),
            'cls': torch.tensor(class_labels, dtype=torch.float32),
            'bboxes': torch.tensor(norm_bboxes, dtype=torch.float32).reshape(-1, 4)
        }

import torch.nn as nn

@MODEL_REGISTRY.register('UltralyticsClsAdapter')
class UltralyticsClsAdapter(BaseModelAdapter):
    def __init__(self, yaml_model: str = "yolov8m-cls.pt", num_classes: int = 2, loss: dict = None):
        super().__init__()
        from ultralytics import YOLO
        
        _yolo = YOLO(yaml_model)
        self.model = _yolo.model
        
        # Ultralytics PT loads freeze the entire network by default, disabling backbone learning.
        # We explicitly unfreeze all layers for end-to-end task finetuning in CellStudio.
        for p in self.model.parameters():
            p.requires_grad = True
        
        # Override classification head if mismatch
        if getattr(self.model, 'nc', None) != num_classes and hasattr(self.model, 'model') and hasattr(self.model.model[-1], 'linear'):
            in_features = self.model.model[-1].linear.in_features
            self.model.model[-1].linear = nn.Linear(in_features, num_classes)
            self.model.nc = num_classes

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
        class_weights = loss.get('class_weights') if loss else None
        weights = torch.tensor(class_weights, dtype=torch.float).cuda() if class_weights and torch.cuda.is_available() else \
                  torch.tensor(class_weights, dtype=torch.float) if class_weights else None
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)

    def _extract_labels(self, data_samples: list):
        if data_samples is None: return None
        labels = []
        for sample in data_samples:
            if isinstance(sample, dict):
                if 'gt_labels' in sample:
                    labels.append(sample['gt_labels'][0] if isinstance(sample['gt_labels'], (list, np.ndarray, torch.Tensor)) else sample['gt_labels'])
                else:
                    labels.append(sample.get('gt_label', 0))
            else:
                if sample.gt_instances is not None and sample.gt_instances.labels is not None:
                    labels.append(sample.gt_instances.labels[0].item())
                else:
                    labels.append(0)
        device = next(self.model.parameters()).device
        return torch.tensor(labels, dtype=torch.long, device=device)

    def forward_train(self, imgs: torch.Tensor, data_samples: list = None, **kwargs):
        device = next(self.model.parameters()).device
        imgs = imgs.to(device)
        logits = self.model(imgs)
        gt_labels = self._extract_labels(data_samples)
        
        if gt_labels is not None:
            loss = self.loss_fn(logits, gt_labels)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = (preds == gt_labels).float().mean().item() if gt_labels is not None else 0.0
            
        return {'loss': loss, 'loss_cls': loss.item(), 'acc': acc}
        
    def forward_test(self, imgs: torch.Tensor, data_samples: list = None, **kwargs):
        device = next(self.model.parameters()).device
        imgs = imgs.to(device)
        logits_out = self.model(imgs)
        
        if isinstance(logits_out, (list, tuple)):
            probs = logits_out[0]
            logits = logits_out[1]
        else:
            logits = logits_out
            probs = torch.softmax(logits, dim=1)
            
        preds = torch.argmax(probs, dim=1)
        
        gt_labels = self._extract_labels(data_samples) if data_samples else None
        
        val_loss = 0.0
        if gt_labels is not None:
            val_loss = self.loss_fn(logits, gt_labels).item()
            
        return {
            'loss_cls': val_loss,
            'probs': probs.detach().cpu(),
            'preds': preds.detach().cpu(),
            'gt_labels': gt_labels.detach().cpu() if gt_labels is not None else None
        }
