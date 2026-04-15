import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from typing import List, Dict, Optional

from .base import BaseModelAdapter
from ..builder import MODEL_REGISTRY
from ...structures.results import DataSample, CellStudioInferResult

@MODEL_REGISTRY.register('SMPAdapter')
class SMPAdapter(BaseModelAdapter):
    """
    Adapter for Segmentation Models Pytorch (SMP) architectures (UNet, DeepLabV3, etc.)
    Provides integration for semantic training and watershed-based instance extraction during inference.
    """
    def __init__(self, arch: str, encoder_name: str, encoder_weights: str = "imagenet", in_channels: int = 3, classes: int = 1):
        super().__init__()
        self.arch = arch
        self.num_classes = classes
        
        if arch.lower() == 'unet':
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
            )
        elif arch.lower() == 'deeplabv3':
            self.model = smp.DeepLabV3(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
            )
        elif arch.lower() == 'deeplabv3plus':
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
            )
        else:
            raise ValueError(f"Unsupported SMP architecture: {arch}")
            
        # Losses
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def dice_loss(self, pred, target, smooth=1e-5):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1 - (2. * intersection + smooth) / (union + smooth)

    def forward_train(self, imgs: torch.Tensor, data_samples: Optional[List[DataSample]]) -> Dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        imgs = imgs.to(device)
        logits = self.model(imgs) # [B, 1, H, W]
        
        # Assemble batched ground truth
        targets = []
        for ds in data_samples:
            if hasattr(ds, 'gt_semantic_seg'):
                targets.append(ds.gt_semantic_seg) # [H, W]
        
        target_tensor = torch.stack(targets).unsqueeze(1).float() # [B, 1, H, W]
        target_tensor = target_tensor.to(logits.device)
        
        # Calculate BCE and Dice
        bce = self.bce_loss(logits, target_tensor)
        dice = self.dice_loss(logits, target_tensor)
        
        loss = bce + dice
        
        return {
            'loss': loss,
            'bce_loss': bce,
            'dice_loss': dice
        }

    def _extract_instances_watershed(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Marker-controlled watershed or simple connected components 
        to divide semantic masks into instance masks.
        """
        # Distance transform
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        # Threshold to get confident markers
        ret, markers = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        markers = np.uint8(markers)
        
        # Connected components for markers
        ret, markers = cv2.connectedComponents(markers)
        markers = markers + 1
        
        # Unknown regio (things that are mask but not marker)
        unknown = cv2.subtract(binary_mask, np.uint8(markers > 1) * 255)
        markers[unknown == 255] = 0
        
        # Watershed
        # cv2.watershed expects 3 channel image. We synthesize one.
        img_synth = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_synth, markers)
        
        instance_mask = np.zeros_like(binary_mask, dtype=np.int32)
        instance_mask[markers > 1] = markers[markers > 1] - 1
        
        return instance_mask

    def forward_test(self, imgs: torch.Tensor, data_samples: Optional[List[DataSample]] = None) -> List[CellStudioInferResult]:
        device = next(self.model.parameters()).device
        imgs = imgs.to(device)
        logits = self.model(imgs) # [B, 1, H, W]
        probs = torch.sigmoid(logits)
        
        results = []
        for i in range(imgs.size(0)):
            prob_map = probs[i, 0].cpu().numpy()
            binary_mask = (prob_map > 0.5).astype(np.uint8) * 255
            
            # Post-processing: extract instances
            instance_map = self._extract_instances_watershed(binary_mask)
            
            # Convert instance_map (H, W) -> individual bounding boxes and masks
            instance_ids = np.unique(instance_map)
            instance_ids = instance_ids[instance_ids > 0]
            
            masks = []
            bboxes = []
            labels = []
            scores = []
            
            for inst_id in instance_ids:
                inst_mask = (instance_map == inst_id).astype(np.uint8)
                
                # Fetch bbox from mask
                y_indices, x_indices = np.where(inst_mask > 0)
                if len(y_indices) == 0:
                    continue
                xmin, xmax = x_indices.min(), x_indices.max()
                ymin, ymax = y_indices.min(), y_indices.max()
                
                # Filter extremely small artifacts
                if (xmax - xmin) < 3 or (ymax - ymin) < 3:
                    continue
                    
                bboxes.append([xmin, ymin, xmax, ymax])
                masks.append(torch.from_numpy(inst_mask).bool())
                labels.append(0) # single class
                
                # Estimate confidence from probe map average over mask
                score = prob_map[inst_mask > 0].mean()
                scores.append(float(score))
                
            res = CellStudioInferResult()
            if len(bboxes) > 0:
                res.bboxes = torch.tensor(bboxes, dtype=torch.float32).to(imgs.device)
                res.masks = torch.stack(masks).to(imgs.device)
                res.labels = torch.tensor(labels, dtype=torch.long).to(imgs.device)
                res.scores = torch.tensor(scores, dtype=torch.float32).to(imgs.device)
            else:
                res.bboxes = torch.zeros((0, 4), dtype=torch.float32).to(imgs.device)
                res.masks = torch.zeros((0, prob_map.shape[0], prob_map.shape[1]), dtype=torch.bool).to(imgs.device)
                res.labels = torch.zeros((0,), dtype=torch.long).to(imgs.device)
                res.scores = torch.zeros((0,), dtype=torch.float32).to(imgs.device)
                
            results.append(res)
            
        return results
