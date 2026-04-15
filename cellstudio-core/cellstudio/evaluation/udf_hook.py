import os
import copy
from typing import Any, Dict, List

from ..structures.csuos import (
    CSUFeatureCollection, CSUFeature, CSUProperties, 
    CSUGeometry, CSUClassProperties, CSUMetadata, CSUImageContext
)

class UDFSerializer:
    """Orchestrates the conversion of standard inferential dictionary outputs into UDF JSON.
    
    This acts as the bridge for creating multi-model cascaded networks:
    Outputs dumped here can directly be piped as inputs to the UDFDataset.
    """
    
    @staticmethod
    def dump_predictions_to_udf(
        work_dir: str, 
        y_true: Any, 
        y_pred: Any, 
        y_prob: Any, 
        data_samples: List[Dict] = None
    ) -> None:
        """Serialize ML predictions to a UDF compliant JSON file.
        
        Args:
            work_dir: Directory to save ``udf_predictions.json``.
            y_true: Ground-truth data.
            y_pred: Prediction data (Bounding boxes, Labels, Masks, etc).
            y_prob: Optional probability/confidence array.
            data_samples: The metadata mapped via the dataset for resolving dimensions.
        """
        if not data_samples:
            return  # Need Image ID bindings to build UDF
            
        features = []
        image_contexts = []
        
        # Assumption: For detection, y_pred is list of shape (N, 4) bboxes and labels
        # For Classification, y_pred is list of labels
        
        for idx, sample in enumerate(data_samples):
            img_id = sample.get('image_id', f"img_{idx}")
            img_path = sample.get('img_path', '')
            width = sample.get('width', 0)
            height = sample.get('height', 0)
            
            ctx = CSUImageContext(image_id=img_id, file_path=img_path, width=width, height=height)
            if ctx not in image_contexts:
                image_contexts.append(ctx)
                
            prediction = y_pred[idx] if idx < len(y_pred) else None
            probability = y_prob[idx] if y_prob is not None and idx < len(y_prob) else None
            
            # Simple heuristic detection extraction:
            if isinstance(prediction, dict):
                # Typically detection dumps {'bboxes': [], 'labels': [], 'scores': []}
                bboxes = prediction.get('bboxes', [])
                labels = prediction.get('labels', [])
                scores = prediction.get('scores', [])
                
                for b_idx, box in enumerate(bboxes):
                    geom = CSUGeometry(type="BBox", coordinates=list(box))
                    
                    lbl_id = labels[b_idx] if b_idx < len(labels) else 0
                    score = float(scores[b_idx]) if b_idx < len(scores) else float(probability) if probability is not None else 1.0
                    
                    props = CSUProperties(
                        object_type="detection",
                        classification=CSUClassProperties(name=f"Class_{lbl_id}", id=lbl_id),
                        measurements={"confidence": score}
                    )
                    
                    feat = CSUFeature(
                        geometry=geom, properties=props, image_id=img_id,
                        id=f"pred_{img_id}_{b_idx}"
                    )
                    features.append(feat)
                    
            elif isinstance(prediction, (int, float)):
                # Classification Global Label
                conf = float(probability) if probability is not None else 1.0
                geom = CSUGeometry(type="BBox", coordinates=[0, 0, width, height])
                props = CSUProperties(
                    object_type="image",
                    classification=CSUClassProperties(name=f"Class_{prediction}", id=int(prediction)),
                    measurements={"confidence": conf}
                )
                feat = CSUFeature(geometry=geom, properties=props, image_id=img_id, id=f"cls_{img_id}")
                features.append(feat)
                
        # Generate the master file
        if features:
            meta = CSUMetadata(generator="UDF_Infer_Hook_v1")
            collection = CSUFeatureCollection(features=features, images=image_contexts, metadata=meta)
            
            os.makedirs(work_dir, exist_ok=True)
            collection.save_json(os.path.join(work_dir, 'udf_predictions.json'))
