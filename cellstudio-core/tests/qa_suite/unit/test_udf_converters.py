import os
import json
import pytest
import numpy as np

from cellstudio.structures.csuos import CSUFeatureCollection, CSUFeature, CSUGeometry, CSUProperties, CSUImageContext
from cellstudio.datasets.converters.coco import COCOConverter
from cellstudio.datasets.converters.yolo import YOLOConverter
from cellstudio.datasets.udf_dataset import UDFDataset

@pytest.fixture
def dummy_udf_collection():
    """Generates a dummy CSUFeatureCollection for bidirectional testing."""
    geom = CSUGeometry(type="BBox", coordinates=[10, 20, 100, 200])
    props = CSUProperties(object_type="detection")
    props.classification = type('obj', (object,), {'id': 1, 'name': 'Tumor', 'color': [255,0,0]})()
    
    feat = CSUFeature(geometry=geom, properties=props, image_id="img_001", id="feat_1")
    
    img_ctx = CSUImageContext(image_id="img_001", file_path="test_img.jpg", width=500, height=500)
    
    return CSUFeatureCollection(features=[feat], images=[img_ctx])


def test_coco_converter_bidirectional(tmp_path, dummy_udf_collection):
    """Product-level test ensuring COCO conversions do not leak or misalign data."""
    converter = COCOConverter()
    
    # Export UDF -> COCO
    coco_path = os.path.join(tmp_path, 'test_coco.json')
    converter.export_from_udf(dummy_udf_collection, coco_path)
    assert os.path.exists(coco_path)
    
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)
        
    assert len(coco_data['images']) == 1
    assert len(coco_data['annotations']) == 1
    assert coco_data['annotations'][0]['bbox'] == [10, 20, 90, 180] # width: 90, height: 180
    
    # Import COCO -> UDF
    parsed_udf = converter.parse_to_udf(coco_path)
    assert len(parsed_udf.features) == 1
    assert parsed_udf.features[0].image_id == "img_001"
    # Ensure back to absolute coordinates
    assert parsed_udf.features[0].geometry.coordinates == [10, 20, 100, 200]
    
def test_yolo_converter_bidirectional(tmp_path, dummy_udf_collection):
    """Product-level test ensuring YOLO normalizations map correctly."""
    converter = YOLOConverter()
    
    # Setup test dir
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    
    # Dummy image to simulate directory structure for YOLO import
    open(img_dir / "img_001.jpg", 'w').close()
    
    # Export UDF -> YOLO (Expects normalized)
    converter.export_from_udf(dummy_udf_collection, str(lbl_dir))
    
    txt_path = lbl_dir / "img_001.txt"
    assert txt_path.exists()
    
    with open(txt_path, 'r') as f:
        yolo_str = f.read().strip().split(' ')
        # Should be Normalized based on 500x500. [x_c, y_c, w, h]
        # xmin=10, xmax=100 -> x_c = 55 -> norm 0.11
        assert yolo_str[0] == "1" # class id
        assert pytest.approx(float(yolo_str[1])) == 0.11 

def test_udf_dataset_ingestion(tmp_path, dummy_udf_collection):
    """Test standard UDFDataset Dataloader format creation."""
    json_path = os.path.join(tmp_path, "udf_annotations.json")
    dummy_udf_collection.save_json(json_path)
    
    dataset = UDFDataset(data_root=str(tmp_path), ann_file="udf_annotations.json", pipeline=None)
    
    # Dataset should group features by image
    assert len(dataset) == 1
    
    item = dataset[0]
    assert item['image_id'] == "img_001"
    assert len(item['gt_bboxes']) == 1
    assert item['gt_labels'][0] == 1
