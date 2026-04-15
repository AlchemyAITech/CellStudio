import os
import yaml

TARGET_DIR = "tests/integration/configs"

def generate_yaml(filename, base_config, dataset_yaml, task_type, model_block, optim_block=None):
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # We embed the configurations with _base_ tags to reuse existing base definitions
    out = {
        '_base_': [
            'base_tiny_runtime.yaml'
        ],
        'task': {'type': task_type},
        'model': model_block
    }
    
    if optim_block:
        out['optim_wrapper'] = optim_block
        
    path = os.path.join(TARGET_DIR, filename)
    
    # Custom dump because PyYAML default doesn't handle _base_ beautifully
    with open(path, 'w') as f:
        f.write(f"_base_:\n  - base_tiny_runtime.yaml\n  - {dataset_yaml}\n\n")
        f.write(yaml.dump({'task': {'type': task_type}}, default_flow_style=False))
        f.write("\n")
        f.write(yaml.dump({'model': model_block}, default_flow_style=False))
        if optim_block:
            f.write("\n")
            f.write(yaml.dump({'optim_wrapper': optim_block}, default_flow_style=False))
            
    print(f"Generated: {path}")

def main():
    print("Generating Full-Scale Test Matrix Configurations...")
    
    # Ensure datasets exist, we use the tiny datasets defined in the project which are fast to execute!
    # --- Classification ---
    cls_models = {
        "tiny_cls_resnet18.yaml": {'type': 'TimmAdapter', 'model_name': 'resnet18', 'pretrained': False, 'num_classes': 2},
        "tiny_cls_effnet.yaml": {'type': 'TimmAdapter', 'model_name': 'efficientnet_b0', 'pretrained': False, 'num_classes': 2},
        "tiny_cls_vit.yaml": {'type': 'TimmAdapter', 'model_name': 'vit_base_patch16_224', 'pretrained': False, 'num_classes': 2},
        "tiny_cls_yolov8.yaml": {'type': 'UltralyticsAdapter', 'yaml_model': 'yolov8n-cls.yaml', 'pretrained': False, 'num_classes': 2},
        "tiny_cls_yolo26.yaml": {'type': 'UltralyticsAdapter', 'yaml_model': 'yolo26n-cls.yaml', 'pretrained': False, 'num_classes': 2}
    }
    
    for fname, model_blk in cls_models.items():
        base_ds = "base_tiny_cls_dataset.yaml" # Assuming defined or we create inline!
        # Actually our tiny pipelines merge dataset entirely inside the yaml!
        # Let's write them completely inline for tests!
        pass 
        
    # Since generating 17 fragmented YAMLs with duplicate datasets is messy, 
    # Let's dynamically create them in a structured way!
    
    cls_ds = {
        'type': 'UDFDataset',
        'data_root': 'datasets/classfication/MIDOG_tiny',  # Intentionally misspelled as per directory on disk
        'ann_file': 'udf_train.json',
        'task_filter': 'image',
        'pipeline': [
            {'type': 'LoadImageFromFile'},
            {'type': 'Resize', 'size': [224, 224]},
            {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]},
            {'type': 'PackCellStudioInputs'}
        ]
    }
    
    det_ds = {
        'type': 'UDFDataset',
        'data_root': 'datasets/detection/MIDO_tiny',
        'ann_file': 'udf_standard.json',
        'task_filter': 'detection',
        'pipeline': [
            {'type': 'LoadImageFromFile'},
            {'type': 'Resize', 'size': [640, 640]},
            {'type': 'Normalize', 'mean': [0, 0, 0], 'std': [255.0, 255.0, 255.0]},
            {'type': 'PackCellStudioInputs'}
        ]
    }
    
    seg_ds = {
        'type': 'UDFDataset',
        'data_root': 'datasets/segmentation/cellpose_tiny',
        'ann_file': 'udf_train.json',
        'task_filter': 'cell',
        'pipeline': [
            {'type': 'LoadImageFromFile'},
            {'type': 'Resize', 'size': [512, 512]},
            {'type': 'Normalize', 'mean': [0, 0, 0], 'std': [255.0, 255.0, 255.0]},
            {'type': 'PackCellStudioInputs'}
        ]
    }
    
    def write_yaml(fname, task_str, ds_block, model_block, metrics, lr=0.001):
        path = os.path.join(TARGET_DIR, fname)
        cfg = {
            '_base_': ['base_tiny_runtime.yaml'],
            'task': {'type': task_str},
            'model': model_block,
            'train_dataloader': {'batch_size': 2, 'num_workers': 0, 'dataset': ds_block},
            'val_dataloader': {'batch_size': 2, 'num_workers': 0, 'dataset': ds_block},
            'val_evaluator': {'metrics': metrics},
            'optim_wrapper': {'optimizer': {'type': 'AdamW', 'lr': lr}}
        }
        with open(path, 'w') as f:
            yaml.dump(cfg, f, sort_keys=False)
        print(f"-> {fname}")

    # ===== Classification =====
    cls_metrics = [{'type': 'Accuracy'}, {'type': 'Precision'}, {'type': 'Recall'}, {'type': 'F1_Score'}]
    write_yaml('tiny_cls_resnet18.yaml', 'ClassificationTask', cls_ds, {'type': 'TimmClassifier', 'architecture': 'resnet18', 'pretrained': False, 'num_classes': 2}, cls_metrics)
    write_yaml('tiny_cls_vit.yaml', 'ClassificationTask', cls_ds, {'type': 'TimmClassifier', 'architecture': 'vit_base_patch16_224', 'pretrained': False, 'num_classes': 2}, cls_metrics)
    write_yaml('tiny_cls_yolo26.yaml', 'ClassificationTask', cls_ds, {'type': 'UltralyticsClsAdapter', 'yaml_model': 'yolov8n-cls.yaml', 'num_classes': 2}, cls_metrics)
    
    # ===== Detection =====
    det_metrics = [{'type': 'det_map50'}, {'type': 'det_precision'}, {'type': 'det_recall'}, {'type': 'det_f1'}, {'type': 'det_count_error'}]
    write_yaml('tiny_det_retinanet.yaml', 'ObjectDetectionTask', det_ds, {'type': 'MMDetAdapter', 'config_file': 'configs/detect/mmdet_configs/rtmdet_m_8xb32-300e_coco.py', 'pretrained': '', 'num_classes': 2}, det_metrics)
    write_yaml('tiny_det_yolo26.yaml', 'ObjectDetectionTask', det_ds, {'type': 'UltralyticsDetAdapter', 'yaml_model': 'yolov8n.yaml', 'pretrained': False, 'num_classes': 2}, det_metrics)
    
    # ===== Segmentation =====
    seg_metrics = [{'type': 'seg_miou'}, {'type': 'seg_dice'}, {'type': 'seg_hd95'}, {'type': 'seg_pq'}, {'type': 'seg_aji'}, {'type': 'seg_all_metrics'}]
    write_yaml('tiny_seg_maskrcnn.yaml', 'InstanceSegmentationTask', seg_ds, {'type': 'MMDetAdapter', 'config_file': 'configs/detect/mmdet_configs/faster-rcnn_r50_fpn_1x_coco.py', 'pretrained': '', 'num_classes': 2}, seg_metrics)
    write_yaml('tiny_seg_yolo26.yaml', 'InstanceSegmentationTask', seg_ds, {'type': 'UltralyticsSegAdapter', 'yaml_model': 'yolov8n-seg.yaml', 'pretrained': False, 'num_classes': 2}, seg_metrics)
    write_yaml('tiny_seg_cpsam.yaml', 'InstanceSegmentationTask', seg_ds, {'type': 'CellposeSAMAdapter'}, seg_metrics)

if __name__ == '__main__':
    main()
