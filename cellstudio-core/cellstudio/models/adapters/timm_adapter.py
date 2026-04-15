import torch
import torch.nn as nn
import numpy as np
from ..builder import MODEL_REGISTRY

@MODEL_REGISTRY.register('TimmClassifier')
class TimmClassifier(nn.Module):
    """
    Standard Architecture for TIMM backends.
    """
    def __init__(self, architecture: str = 'resnet18', num_classes: int = 2, pretrained: bool = True, loss: dict = None):
        super().__init__()
        import timm
        try:
            self.model = timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)
        except Exception as e:
            raise RuntimeError(f"Failed to create timm model {architecture}: {str(e)}")
            
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
        class_weights = loss.get('class_weights') if loss else None
        weights = torch.tensor(class_weights, dtype=torch.float).cuda() if class_weights and torch.cuda.is_available() else \
                  torch.tensor(class_weights, dtype=torch.float) if class_weights else None
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        
    def _extract_labels(self, data_samples: list):
        if not data_samples:
             raise ValueError("data_samples cannot be None")
        labels = []
        for sample in data_samples:
            if isinstance(sample, dict):
                # Legacy pipeline support
                if 'gt_labels' in sample:
                    labels.append(sample['gt_labels'][0] if isinstance(sample['gt_labels'], (list, np.ndarray, torch.Tensor)) else sample['gt_labels'])
                else:
                    labels.append(sample.get('gt_label', 0))
            else:
                # PackCellStudioInputs Object support
                if sample.gt_instances is not None and sample.gt_instances.labels is not None:
                    labels.append(sample.gt_instances.labels[0].item())
                else:
                    labels.append(0)  # Default fallback if empty
                
        device = next(self.model.parameters()).device
        return torch.tensor(labels, dtype=torch.long, device=device)

    def forward_train(self, imgs: torch.Tensor, data_samples: list = None, **kwargs):
        device = next(self.model.parameters()).device
        imgs = imgs.to(device)
        logits = self.model(imgs)
        gt_labels = self._extract_labels(data_samples)
        
        loss = self.loss_fn(logits, gt_labels)
        
        # Calculate Train Accuracy
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = (preds == gt_labels).float().mean().item()
        
        return {'loss': loss, 'loss_cls': loss.item(), 'acc': acc}
        
    def forward_test(self, imgs: torch.Tensor, data_samples: list = None, **kwargs):
        device = next(self.model.parameters()).device
        imgs = imgs.to(device)
        logits = self.model(imgs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        gt_labels = self._extract_labels(data_samples)
        
        val_loss = 0.0
        if gt_labels is not None:
            val_loss = self.loss_fn(logits, gt_labels).item()
        
        return {
            'loss_cls': val_loss,
            'probs': probs.detach().cpu(),
            'preds': preds.detach().cpu(),
            'gt_labels': gt_labels.detach().cpu()
        }

    # Failsafe forward
    def forward(self, imgs, mode='tensor', **kwargs):
        return self.model(imgs)
