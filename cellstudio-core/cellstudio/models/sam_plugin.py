import os
import numpy as np
from typing import List, Dict, Any, Union

class SAMAutoAnnotator:
    """
    Zero-shot/Few-shot Segment Anything Model (SAM) Plugin for PathStudio.
    Exposes a unified interface for the frontend to pass point clicks or bounding boxes
    and retrieve high-quality polygon masks for auto-annotation.
    """
    
    def __init__(self, model_type: str = "mobile_sam", device: str = "cpu"):
        """
        model_type: 'mobile_sam' (faster, edge-friendly) or 'sam_b', 'sam_l'
        """
        self.model_type = model_type
        self.device = device
        self.model = self._load_model()
        
    def _load_model(self):
        try:
            from ultralytics import SAM
        except ImportError:
            raise ImportError("Ultralytics library required for SAM. `pip install ultralytics`")
            
        # Support light-weight SAM for interactive client use
        model_map = {
            "mobile_sam": "mobile_sam.pt",
            "sam_b": "sam_b.pt",
            "sam_l": "sam_l.pt"
        }
        weights = model_map.get(self.model_type, "mobile_sam.pt")
        print(f"[SAM Plugin] Loading {weights} on {self.device}...")
        return SAM(weights)
        
    def predict_with_prompts(
        self, 
        image_source: Union[str, np.ndarray], 
        points: List[List[int]] = None, 
        labels: List[int] = None,
        bboxes: List[List[int]] = None
    ) -> List[Dict]:
        """
        Predict masks using user prompts (points or boxes).
        
        Args:
            image_source: Path to image or cv2/numpy array (H, W, C).
            points: List of [x, y] coordinates. e.g., [[150, 200], [180, 220]]
            labels: List of 1 (foreground point) or 0 (background point). e.g., [1, 0]
            bboxes: List of bounding boxes [x1, y1, x2, y2].
            
        Returns:
            A list of dictionary containing {"mask": np.array, "polygon": list, "score": float}
        """
        predict_args = {
            "source": image_source,
            "device": self.device,
            "retina_masks": True, # High resolution masks
            "stream": False
        }
        
        if points and labels:
            predict_args["points"] = points
            predict_args["labels"] = labels
            
        if bboxes:
            predict_args["bboxes"] = bboxes
            
        results = self.model.predict(**predict_args)
        
        output = []
        for r in results:
            if r.masks is not None:
                for i, (mask, conf) in enumerate(zip(r.masks.data, r.boxes.conf)):
                    # Convert mask to polygon for frontend rendering compatibility (deck.gl / OSD)
                    poly = r.masks.xy[i].tolist()
                    output.append({
                        "polygon": poly,
                        "score": float(conf),
                        "mask_shape": mask.shape
                    })
        return output
