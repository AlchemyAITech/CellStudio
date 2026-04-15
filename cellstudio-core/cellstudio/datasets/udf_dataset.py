import json
import os
from typing import Any, Dict, List, Optional

from .base import BaseDataset
from .registry import DatasetRegistry
from ..structures.csuos import CSUFeatureCollection, CSUFeature, CSUImageContext

@DatasetRegistry.register('UDFDataset')
class UDFDataset(BaseDataset):
    """The Universal Data Foundation Dataset.
    
    This unified dataset dynamically adapts its output based on the types 
    of features stored in the Universal CSV/JSON representation. 
    It replaces individual ClassificationDataset, DetectionDataset, and SegmentationDataset.
    """
    
    def __init__(
        self,
        data_root: str,
        ann_file: str,
        data_prefix: Optional[Dict[str, str]] = None,
        pipeline: Optional[List[Dict]] = None,
        test_mode: bool = False,
        task_filter: Optional[str] = None # 'classification', 'detection', 'cell'
    ):
        self.task_filter = task_filter
        super().__init__(data_root, ann_file, data_prefix, pipeline, test_mode)

    def _load_data_list(self) -> List[Dict[str, Any]]:
        """Parses the UDF JSON and converts it to a layout suited for the pipelines."""
        assert self.ann_file.endswith('.json'), "UDF annotations must be in JSON format."
        
        with open(self.ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        images = data.get('images', [])
        features = data.get('features', [])
        feature_files = data.get('feature_files', {})
        
        # Build image context map
        img_map = {img['image_id']: img for img in images}
        # Group features by image_id
        feat_map = {}
        
        # Load external linked JSON annotations if applicable
        if feature_files:
            base_dir = os.path.dirname(self.ann_file)
            for img_id, rel_path in feature_files.items():
                target_path = os.path.join(base_dir, rel_path)
                if os.path.exists(target_path):
                    with open(target_path, 'r', encoding='utf-8') as sf:
                        subjson = json.load(sf)
                        feat_map[img_id] = subjson.get('features', [])
        
        for feat in features:
            iid = feat.get('image_id')
            if not iid: continue
            if iid not in feat_map:
                feat_map[iid] = []
            feat_map[iid].append(feat)
            
        # Optional Task Filtering
        if self.task_filter:
            for iid in list(feat_map.keys()):
                filtered = [f for f in feat_map[iid] if f.get('properties', {}).get('object_type') == self.task_filter]
                feat_map[iid] = filtered
            
        data_list = []
        img_prefix = self.data_prefix.get('img_path', '')
        
        # Build classes dynamically if not provided
        if getattr(self, 'CLASSES', None) is None:
            classes = set()
            for f in features:
                cls_name = f.get('properties', {}).get('class', {}).get('name')
                if cls_name: classes.add(cls_name)
            self.CLASSES = sorted(list(classes))
        
        for img_id, context in img_map.items():
            file_name = context.get('file_path', '')
            if file_name and ('/' in file_name or '\\' in file_name):
                # If it's an absolute path from Mac (/Users/) or standard absolute path, just take the basename.
                # However, if it's a relative path like 'images/8185.png', we MUST keep 'images/8185.png'.
                if file_name.startswith('/Users/') or file_name.startswith('C:') or file_name.startswith('D:') or file_name.startswith('E:'):
                    file_name = os.path.basename(file_name)
                    
            img_path = os.path.join(self.data_root, img_prefix, file_name)
            if not os.path.exists(img_path):
                # Fallback: legacy dataset sometimes dumps "341.tiff" but actually it is in "images/341.tiff"
                fallback_path = os.path.join(self.data_root, img_prefix, 'images', file_name)
                if os.path.exists(fallback_path):
                    img_path = fallback_path
                else:
                    # Skip silently, preventing pipeline crash on missing legacy assets
                    continue
            
            # Form standard dict recognized by BaseDataset and Transform pipelines
            data_info = {
                'img_path': img_path,
                'image_id': img_id,
                'width': context.get('width', 0),
                'height': context.get('height', 0),
                'udf_features': feat_map.get(img_id, []) # The raw dict dictionaries of CSUFeatures
            }
            
            # Pre-parse common attributes to avoid writing custom pipelines for everything
            labels = []
            bboxes = []
            masks = []
            
            for f in data_info['udf_features']:
                props = f.get('properties', {})
                geom = f.get('geometry', {})
                cls_info = props.get('class', {})
                
                cls_name = cls_info.get('name')
                if cls_name and hasattr(self, 'CLASSES') and cls_name in self.CLASSES:
                    labels.append(self.CLASSES.index(cls_name))
                else:
                    labels.append(cls_info.get('id', 0))
                
                # Check geometry for BBox or Polygon
                if geom.get('type') == 'BBox':
                    bboxes.append(geom.get('coordinates', [0,0,0,0]))
                elif geom.get('type') == 'Polygon':
                    # Extract bounding box from Polygon (for hybrid tasks)
                    poly = geom.get('coordinates', [[[]]])[0]
                    if poly and len(poly) > 0:
                        xs = [p[0] for p in poly]
                        ys = [p[1] for p in poly]
                        bboxes.append([min(xs), min(ys), max(xs), max(ys)])
                    else:
                        bboxes.append([0.0, 0.0, 1.0, 1.0]) # Fallback for corrupted empty polygons
                    masks.append(geom.get('coordinates', []))
                    
            if bboxes:
                import numpy as np
                data_info['gt_bboxes'] = np.array(bboxes, dtype=np.float32)
            if labels:
                import numpy as np
                data_info['gt_labels'] = np.array(labels, dtype=np.int64)
            if masks:
                data_info['gt_masks'] = masks # leave masks as list of polys for downstream dict mapping
                
            data_list.append(data_info)
            
        return data_list
