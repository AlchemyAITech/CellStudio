from typing import Any, Dict, List

import numpy as np


class CascadePipeline:
    """
    Executes a Direct Acyclic Graph (DAG) of AI models for Medical Pathology tasks.
    E.g.: [Segment Tissue] -> Crop Regions -> [Detect Cells] -> Crop Cells -> [Classify Cells]
    """
    
    def __init__(self, steps: List[Dict]):
        """
        steps is a list of node configurations.
        Example:
        [
            {"name": "Tissue Detector", "model": YoloAdapter(...), "task": "detection", "output_key": "roi_boxes"},
            {"name": "Cell Classifier", "model": TimmAdapter(...), "task": "classification", "input_key": "roi_boxes"}
        ]
        """
        self.steps = steps
        
    def _crop_regions(self, image: np.ndarray, boxes: List[List[int]]) -> List[np.ndarray]:
        """Utility to crop ROIs from an image given bounding boxes."""
        crops = []
        h_img, w_img = image.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            if x2 > x1 and y2 > y1:
                crops.append(image[y1:y2, x1:x2])
        return crops

    def execute(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Runs the cascaded pipeline sequentially.
        """
        context = {"raw_image": image}
        
        for step in self.steps:
            model = step["model"]
            task_type = step["task"]
            name = step["name"]
            
            print(f"[Cascade] Running: {name} ({task_type})")
            
            # Decide input
            input_key = step.get("input_key")
            if not input_key:
                inputs = [image] # Default full image
            else:
                roi_boxes = context[input_key]
                inputs = self._crop_regions(image, roi_boxes)
                
            # Execute Model
            step_outputs = []
            for inp in inputs:
                # We assume model.predict expects a single image and returns a standard dict
                res = model.predict(inp)
                step_outputs.append(res)
                
            # Format and save output to context
            output_key = step.get("output_key", f"{name}_output")
            
            # Simple heuristic mapping for prototype
            if task_type == "detection":
                # Assuming YOLO output format wrapped in results
                all_boxes = []
                for res in step_outputs:
                    for r in res: # batch result
                        if r.boxes:
                            all_boxes.extend(r.boxes.xyxy.cpu().numpy().tolist())
                context[output_key] = all_boxes
                
            elif task_type == "classification":
                # Assuming top-1 class from Timm adapter
                all_classes = []
                for res in step_outputs:
                    # Generic format extraction
                    cls_idx = res.get("top1_idx", -1) 
                    all_classes.append(cls_idx)
                context[output_key] = all_classes
                
        return context

