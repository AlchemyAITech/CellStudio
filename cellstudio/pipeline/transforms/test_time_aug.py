from typing import Dict, Any, List
import copy
from ..registry import PIPELINE_REGISTRY

@PIPELINE_REGISTRY.register('MultiScaleFlipAug')
class MultiScaleFlipAug:
    """
    Test Time Augmentation wrapper.
    Expands a single dictionary node into an aggregated set representing 
    scales and flips, passing them all simultaneously into the final Formatting step.
    """
    def __init__(self, scales: List[tuple], flip: bool = False):
        self.scales = scales
        self.flip = flip

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        aug_results = []
        for scale in self.scales:
            _res = copy.deepcopy(results)
            # Simulated spatial manipulation
            _res['img_shape'] = scale
            
            aug_results.append(_res)
            
            if self.flip:
                _res_flip = copy.deepcopy(_res)
                _res_flip['flip'] = True
                aug_results.append(_res_flip)
                
        # In TTA, the framework usually handles the iteration over variants.
        # This encapsulates the expansion logic gracefully.
        return {'aug_variants': aug_results}
