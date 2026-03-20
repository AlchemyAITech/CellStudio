from typing import Dict, Any
import torch
import numpy as np
from ..registry import PIPELINE_REGISTRY
from ...structures.results import DataSample, InstanceData

@PIPELINE_REGISTRY.register('PackCellStudioInputs')
class PackCellStudioInputs:
    """
    The Zenith Terminus of the Data Pipeline.
    Unforgivingly extracts exactly matched attributes from a raw ResultDict, transferring 
    CPU ndarrays mathematically identically to proper GPU/CPU Torch Tensors.
    Forces outputs into the explicit structural dictionary expected by adapters.
    """
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        packed_results = {}
        
        if 'img' in results:
            img = results['img']
            if isinstance(img, np.ndarray):
                # Standardize to format [C, H, W] if [H, W, C] is passed from loaders like cv2/PIL
                if len(img.shape) == 3 and img.shape[2] <= 4:
                    img = np.transpose(img, (2, 0, 1))
                img = torch.from_numpy(img).float()
            packed_results['imgs'] = img
            
        data_sample = DataSample()
        data_sample.img_path = results.get('img_path', '')
        data_sample.img_shape = results.get('img_shape', (0,0))
        data_sample.ori_shape = results.get('ori_shape', (0,0))
        
        data_sample.metainfo = results.get('img_metas', {})

        gt_instances = InstanceData()
        has_gt = False
        
        if 'gt_bboxes' in results:
            gt_instances.bboxes = torch.from_numpy(results['gt_bboxes']).float()
            has_gt = True
        
        if 'gt_labels' in results:
            gt_instances.labels = torch.from_numpy(results['gt_labels']).long()
            has_gt = True
            
        if 'gt_masks' in results:
            mask_data = results['gt_masks']
            if isinstance(mask_data, np.ndarray):
                mask_data = torch.from_numpy(mask_data).long()
            gt_instances.masks = mask_data
            has_gt = True
            
        if has_gt:
            data_sample.gt_instances = gt_instances
            
        packed_results['data_samples'] = data_sample
        return packed_results
