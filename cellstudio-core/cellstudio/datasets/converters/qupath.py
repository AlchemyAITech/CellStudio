import json
import os
from typing import Dict, Any, List, Optional
from ...structures.csuos import (
    CSUGeometry, CSUClassProperties, CSUProperties, 
    CSUFeature, CSUMetadata, CSUFeatureCollection, CSUImageContext
)
from .base import BaseConverter

class QuPathConverter(BaseConverter):
    """Parser to handle bidirectional conversion between QuPath GeoJSON and CSUOS."""
    
    def parse_to_udf(self, source_path: str, slide_name: Optional[str] = None, **kwargs) -> CSUFeatureCollection:
        filepath = source_path
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if "features" not in data and data.get("type") != "FeatureCollection":
            # Might be a single feature export
            if data.get("type") == "Feature":
                data = {"features": [data]}
            else:
                raise ValueError(f"Invalid QuPath GeoJSON format in {filepath}")
            
        features = []
        for feature in data.get("features", []):
            geom_data = feature.get("geometry", {})
            geom = CSUGeometry(
                type=geom_data.get("type", "Unknown"),
                coordinates=geom_data.get("coordinates", [])
            )
            
            props_data = feature.get("properties", {})
            
            cls_data = props_data.get("classification", None)
            cls_obj = None
            if cls_data:
                cls_obj = CSUClassProperties(
                    name=cls_data.get("name", "Unknown"),
                    color=cls_data.get("color", [255, 0, 0])
                )
                
            # Extract PathCellObject nucleus geometry if present
            nuc_geom = None
            if "nucleusGeometry" in props_data:
                nuc_geom = CSUGeometry(
                    type=props_data["nucleusGeometry"].get("type", "Polygon"),
                    coordinates=props_data["nucleusGeometry"].get("coordinates", [])
                )
                
            props = CSUProperties(
                object_type=props_data.get("objectType", "annotation").lower(),
                classification=cls_obj,
                nucleus_geometry=nuc_geom,
                is_locked=props_data.get("isLocked", False),
                measurements=props_data.get("measurements", {}),
            )
            
            f_obj = CSUFeature(
                geometry=geom,
                properties=props,
                id=feature.get("id"),
                parent_id=feature.get("parent_id") # Map parent relation if QuPath exported it
            )
            features.append(f_obj)
            
        if not slide_name:
            slide_name = os.path.basename(filepath).split('.')[0]
            
        meta = CSUMetadata(
            generator="QuPath_Import"
        )
        # Create a dummy image context for the single slide
        img_ctx = CSUImageContext(
            image_id=slide_name,
            file_path=slide_name,
            width=0, height=0 # We might not know WSI dimensions solely from GeoJSON
        )
        return CSUFeatureCollection(features=features, metadata=meta, images=[img_ctx])

    def export_from_udf(self, udf_collection: CSUFeatureCollection, output_path: str, **kwargs) -> Any:
        # Standard GeoJSON FeatureCollection Dump
        udf_collection.save_json(output_path)

