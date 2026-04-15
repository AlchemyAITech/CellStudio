import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import numpy as np

from .base import BaseModelAdapter
from ..builder import MODEL_REGISTRY
from ...structures.results import DataSample, CellStudioInferResult, InstanceData

@MODEL_REGISTRY.register('CellposeSegAdapter')
class CellposeSegAdapter(BaseModelAdapter):
    """
    Adapter for Cellpose V3/V4 framework. 
    Intercepts and bridges Flow Loss and Probability Cellmask Loss directly to Zenith PyTorch Trainer.
    """
    def __init__(self, model_type: str = "cyto3", pretrained_model: str = None, diam_mean: float = 30.0):
        super().__init__()
        import logging
        import os
        from cellpose import models
        import cellpose.core
        logger = logging.getLogger(__name__)
        
        # Intercept global cellpose model dir
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained_weights")
        cellpose_dir = os.path.join(PRETRAINED_DIR, 'cellpose')
        os.makedirs(cellpose_dir, exist_ok=True)
        models.MODEL_DIR = cellpose_dir
        cellpose.core.MODEL_DIR = cellpose_dir
        
        gpu = torch.cuda.is_available()
        # Initialize CPNet implicitly via CellposeModel wrapper
        if not pretrained_model and model_type:
            pretrained_model = model_type
            
        if pretrained_model:
            self.model = models.CellposeModel(pretrained_model=pretrained_model, gpu=gpu, diam_mean=diam_mean).net
        else:
            self.model = models.CellposeModel(gpu=gpu, diam_mean=diam_mean).net
            
        # Optional SAM module holder
        self.sam_model = None
        
        self.diam_mean = diam_mean
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def _create_flows(self, instance_masks_list):
        # Convert instance masks [H, W] to flows [3, H, W] dynamically per frame
        from cellpose import dynamics
        flows = []
        for mask in instance_masks_list:
            mask_np = np.ascontiguousarray(mask.cpu().numpy())
            if mask_np.max() == 0:
                h, w = mask_np.shape
                flows.append(torch.zeros((3, h, w), dtype=torch.float32))
                continue
                
            # Sanitize broken topologies from Resize(INTER_NEAREST) to prevent Numba infinite loops
            from scipy.ndimage import label
            clean_mask = np.zeros_like(mask_np)
            current_id = 1
            for inst_id in np.unique(mask_np):
                if inst_id == 0: continue
                inst_bin = (mask_np == inst_id)
                labeled_comps, num_comps = label(inst_bin)
                for comp_id in range(1, num_comps + 1):
                    clean_mask[labeled_comps == comp_id] = current_id
                    current_id += 1
                    
            out = dynamics.labels_to_flows([clean_mask])[0]
            flowY, flowX = out[2], out[3]
            prob = out[1]
            
            flow_stack = np.stack([flowY, flowX, prob], axis=0) # [3, H, W]
            flows.append(torch.from_numpy(flow_stack).float())
        return torch.stack(flows)

    def forward_train(self, imgs: torch.Tensor, data_samples: Optional[List[DataSample]]) -> Dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        imgs = imgs.to(device=device, dtype=dtype)
        
        # Cellpose assumes inputs with specific mean/std, but Zenith framework normalizes upstream.
        preds = self.model(imgs) # tuple (y, style)
        y = preds[0] # [B, 3, H, W] (Index 0,1,2: cellprob, flowY, flowX) 
        
        # Extract GT Masks
        mask_list = []
        for ds in data_samples:
            if hasattr(ds, 'gt_instance_seg'):
                mask_list.append(ds.gt_instance_seg)
            else:
                mask_list.append(torch.zeros(imgs.shape[2:], dtype=torch.int32))
                
        # Dynamic Target Flows via Cython core (Run on CPU implicitly within cellpose core)
        target_flows = self._create_flows(mask_list).to(device=device, dtype=dtype)
        
        # In Cellpose `CPnet`, the channel output order is strictly [flowY, flowX, prob] 
        pred_flowY = y[:, 0]
        pred_flowX = y[:, 1]
        pred_prob = y[:, 2]
        
        # Cellpose native scaling: flows are multiplied by 5.0 during training!
        target_flowY = target_flows[:, 0] * 5.0
        target_flowX = target_flows[:, 1] * 5.0
        target_prob = target_flows[:, 2]
        
        loss_flow = self.mse_loss(pred_flowX, target_flowX) + self.mse_loss(pred_flowY, target_flowY)
        loss_prob = self.bce_loss(pred_prob, target_prob) # pred_prob lacks sigmoid if it is logits
        
        loss = 10.0 * loss_flow + loss_prob
        
        return {
            'loss': loss,
            'flow_loss': loss_flow,
            'prob_loss': loss_prob
        }

    def forward_test(self, imgs: torch.Tensor, data_samples: Optional[List[DataSample]] = None) -> List[CellStudioInferResult]:
        from cellpose import dynamics
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        imgs = imgs.to(device=device, dtype=dtype)
        
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            preds = self.model(imgs)
            y = preds[0]
            
        if was_training:
            self.model.train()
            
        y_np = y.to(dtype=torch.float32).cpu().numpy() # [B, 3, H, W]
        
        results = []
        for i in range(len(y_np)):
            # Cellpose models output flows scaled by 5.0 natively. We must revert it before mask computation!
            dP = y_np[i, 0:2] / 5.0 # [2, H, W] (dP[0]=flowY, dP[1]=flowX)
            cellprob = y_np[i, 2]
            
            # During early training (Epoch 1), flows are not perfect strictly to 0.4. Use None to prevent rejecting everything.
            p = dynamics.compute_masks(dP, cellprob, p=None, niter=200, cellprob_threshold=0.0, flow_threshold=None, 
                                       device=torch.device('cpu')) # p is instance mask [H, W]
            
            mask_out = p[0] if isinstance(p, tuple) else p
            res = self._pack_instance_result(mask_out, device)
            results.append(res)
            
        return results
        
    def _pack_instance_result(self, instance_map: np.ndarray, device) -> CellStudioInferResult:
        import numpy as np
        instance_ids = np.unique(instance_map)
        instance_ids = instance_ids[instance_ids > 0]
        
        masks = []
        bboxes = []
        labels = []
        scores = []
        
        for inst_id in instance_ids:
            inst_mask = (instance_map == inst_id).astype(np.uint8)
            y_indices, x_indices = np.where(inst_mask > 0)
            if len(y_indices) == 0:
                continue
            xmin, xmax = x_indices.min(), x_indices.max()
            ymin, ymax = y_indices.min(), y_indices.max()
            
            if (xmax - xmin) < 3 or (ymax - ymin) < 3:
                continue
                
            bboxes.append([xmin, ymin, xmax, ymax])
            masks.append(torch.from_numpy(inst_mask).bool())
            labels.append(0)
            scores.append(1.0) # Flows have no native per-cell confidence
            
        res = CellStudioInferResult()
        if len(bboxes) > 0:
            res.bboxes = torch.tensor(bboxes, dtype=torch.float32).to(device)
            res.masks = torch.stack(masks).to(device)
            res.labels = torch.tensor(labels, dtype=torch.long).to(device)
            res.scores = torch.tensor(scores, dtype=torch.float32).to(device)
        else:
            H, W = instance_map.shape
            res.bboxes = torch.zeros((0, 4), dtype=torch.float32).to(device)
            res.masks = torch.zeros((0, H, W), dtype=torch.bool).to(device)
            res.labels = torch.zeros((0,), dtype=torch.long).to(device)
            res.scores = torch.zeros((0,), dtype=torch.float32).to(device)
        return res

@MODEL_REGISTRY.register('CellposeSAMAdapter')
class CellposeSAMAdapter(CellposeSegAdapter):
    """
    Extensions leveraging the official Cellpose-SAM (CP4) models 
    which wraps the SAM image encoder as its feature extractor.
    Reference: Cellpose Github / cellpose.models.CellposeModel
    """
    def __init__(self, **kwargs):
        kwargs.pop('sam_model_type', None)
        # Force the official SAM-based Cellpose transformer weights
        kwargs['pretrained_model'] = 'cpsam'
        super().__init__(**kwargs)
        # Verify the backbone is indeed SAM-based
        from cellpose.vit_sam import Transformer
        if not isinstance(self.model, Transformer):
            import logging
            logging.warning("Loaded Cellpose model is not the official Transformer(SAM) model!")
