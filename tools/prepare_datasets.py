"""
Prepare UDF Datasets for testing Architecture
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cellstudio-core')))
from cellstudio.structures.csuos import (
    CSUGeometry, CSUProperties, CSUFeature,
    CSUFeatureCollection, CSUMetadata, CSUImageContext
)

def prepare_dummy_udf(output_dir="datasets/dummy_udf"):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'udf_train.json')
    
    # 1. Provide Image Context
    img_context = CSUImageContext(
        image_id="wsi_001",
        file_path="mock_wsi.tiff",
        width=10000,
        height=10000,
        task_type="detection"
    )
    
    # 2. Add properties
    props_annot = CSUProperties(object_type="annotation", source_type="human")

    # 3. Create a parent Region
    parent_tumor = CSUFeature(
        id="region_tumor_01",
        image_id="wsi_001",
        geometry=CSUGeometry(type="Polygon", coordinates=[[[100, 100], [5000, 100], [5000, 5000], [100, 5000]]]),
        properties=props_annot
    )

    # 4. Create child cell nuclei inside the tumor region
    children = []
    for i in range(10):
        bx = 200 + (i * 100)
        by = 200 + (i * 100)
        child = CSUFeature(
            id=f"cell_{i}",
            image_id="wsi_001",
            parent_id="region_tumor_01", # Hierarchy mapped
            geometry=CSUGeometry(type="BBox", coordinates=[bx, by, bx+50, by+50]),
            properties=CSUProperties(object_type="detection", source_type="human")
        )
        children.append(child)
        
    # 5. Pack Collection
    collection = CSUFeatureCollection(
        features=[parent_tumor] + children,
        metadata=CSUMetadata(generator="CellStudio_UDF_Test"),
        images=[img_context]
    )
    
    # Force test hierarchy build
    collection.build_hierarchy_tree()
    assert len(collection.get_children("region_tumor_01")) == 10
    
    # Save standard unified format (with split capabilities if required later)
    collection.save_json(json_path, split_annotations=False, compress_tolerance=0.0)
    print(f"[Success] Generated UDF Dataset at {json_path} successfully mapped with Hierarchy Trees and {len(collection.features)} features.")

if __name__ == '__main__':
    prepare_dummy_udf()
