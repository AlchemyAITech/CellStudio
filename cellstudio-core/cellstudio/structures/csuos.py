import json
import uuid
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class CSUGeometry:
    type: str # "Polygon", "Point", "MultiPolygon", "LineString", "BBox"
    coordinates: List[Any]

@dataclass
class CSUClassProperties:
    name: str
    id: int = 0  # Useful for mapping COCO category_id
    color: List[int] = field(default_factory=lambda: [255, 0, 0])

@dataclass
class CSUProperties:
    """Internal properties tracking what type of ML entity this is."""
    object_type: str # "annotation" (general/masks), "detection" (bounding boxes), "cell" (dual-layer), "image" (whole image classification)
    source_type: str = "human" # "human" (GT) or "algorithm" (Prediction)
    classification: Optional[CSUClassProperties] = None
    nucleus_geometry: Optional[CSUGeometry] = None  # Exclusive for Cell objects requiring dual-ROI
    is_locked: bool = False
    measurements: Dict[str, float] = field(default_factory=dict)
    prompt_hints: Optional[str] = None # For Zero-shot, Few-shot models (Prompt texts)
    match_info: Optional[Dict[str, Any]] = None # Information relating this algorithm pred to human GT, e.g., {'gt_id': str, 'iou': float, 'confidence': float}

@dataclass
class CSUFeature:
    """The fundamental Universal Data Feature for all model inputs/outputs."""
    geometry: CSUGeometry
    properties: CSUProperties
    image_id: Optional[str] = None # Used to link back to a specific image in a batch/COCO
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None  # Flattens tree hierarchy structure for DB/cascade transmission
    type: str = "Feature"
    
    def validate(self):
        assert self.type == "Feature"
        assert self.geometry is not None, "Geometry must be specified"
        assert isinstance(self.properties.object_type, str)

    def to_dict(self) -> Dict:
        """Serialize to standard GeoJSON compliant dictionary."""
        data = {
            "type": self.type,
            "id": self.id,
            "geometry": {
                "type": self.geometry.type,
                "coordinates": self.geometry.coordinates
            },
            "properties": {
                "object_type": self.properties.object_type,
                "source_type": self.properties.source_type,
                "is_locked": self.properties.is_locked,
                "measurements": self.properties.measurements,
            }
        }
        
        if self.image_id:
            data['image_id'] = self.image_id
        if self.parent_id:
            data['parent_id'] = self.parent_id
            
        if self.properties.classification:
            data['properties']['class'] = {
                "name": self.properties.classification.name,
                "id": self.properties.classification.id,
                "color": self.properties.classification.color
            }
            
        if self.properties.nucleus_geometry:
            data['properties']['nucleus_geometry'] = {
                "type": self.properties.nucleus_geometry.type,
                "coordinates": self.properties.nucleus_geometry.coordinates
            }
            
        if self.properties.prompt_hints:
            data['properties']['prompt_hints'] = self.properties.prompt_hints
            
        if self.properties.match_info:
            data['properties']['match_info'] = self.properties.match_info
            
        return data

@dataclass
class CSUMetadata:
    slide_dimensions: Optional[List[int]] = None
    downsample_rate: float = 1.0
    generator: str = "CellStudio_v2_UDF"
    extra_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CSUImageContext:
    """Describes the global properties of an image file to which features belong."""
    image_id: str
    file_path: str
    width: int
    height: int
    task_type: str = "generic" # classification, detection, segmentation, multi_task

@dataclass
class CSUFeatureCollection:
    """The master UDF Dataset Container representing a full dataset or a single image collection."""
    features: List[CSUFeature]
    metadata: CSUMetadata = field(default_factory=CSUMetadata)
    images: List[CSUImageContext] = field(default_factory=list) # To support COCO-like multi-image collections
    type: str = "FeatureCollection"
    
    def validate(self):
        for f in self.features:
            f.validate()

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "metadata": asdict(self.metadata),
            "images": [asdict(img) for img in self.images],
            "features": [f.to_dict() for f in self.features]
        }
        
    def _simplify_coordinates(self, coords, tolerance):
        try:
            from shapely.geometry import Polygon
            # If coordinates are list of loops
            new_coords = []
            for loop in coords:
                if len(loop) < 4:
                    new_coords.append([[round(x, 1), round(y, 1)] for x, y in loop])
                    continue
                poly = Polygon(loop)
                simplified = poly.simplify(tolerance, preserve_topology=False)
                if simplified.geom_type == 'Polygon':
                    new_loop = [[round(x, 1), round(y, 1)] for x, y in list(simplified.exterior.coords)]
                    new_coords.append(new_loop)
                else:
                    new_coords.append([[round(x, 1), round(y, 1)] for x, y in loop])
            return new_coords
        except ImportError:
            # Fallback without RDP
            return [[[round(x, 1), round(y, 1)] for x, y in loop] for loop in coords]

    def save_json(self, path: str, split_annotations: bool = True, compress_tolerance: float = 1.0):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        base_dir = os.path.dirname(path)
        
        # Pre-compress large polygons
        if compress_tolerance > 0:
            for f in self.features:
                if f.geometry.type == 'Polygon':
                    f.geometry.coordinates = self._simplify_coordinates(f.geometry.coordinates, compress_tolerance)
        
        if split_annotations:
            ann_dir = os.path.join(base_dir, 'annotations')
            os.makedirs(ann_dir, exist_ok=True)
            
            # Group features
            feat_map = {}
            for f in self.features:
                iid = f.image_id
                if not iid: continue
                if iid not in feat_map:
                    feat_map[iid] = []
                feat_map[iid].append(f)
                
            index_data = {
                "type": self.type,
                "metadata": asdict(self.metadata),
                "images": [asdict(img) for img in self.images],
                "feature_files": {}
            }
            
            for iid, feats in feat_map.items():
                rel_path = f"annotations/{iid}.json"
                out_path = os.path.join(base_dir, rel_path)
                # Create standard GeoJSON for this specific image
                local_collection = {
                    "type": "FeatureCollection",
                    "features": [f.to_dict() for f in feats]
                }
                with open(out_path, 'w', encoding='utf-8') as lf:
                    json.dump(local_collection, lf, indent=2, ensure_ascii=False)
                index_data["feature_files"][iid] = rel_path
                
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            
    @classmethod
    def load_json(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Loader logic is primarily implemented down in UDFDataset to avoid circular deps and heavy structural rebuilding.
        pass
