import os

base_template = """_base_:
  - ../_base_/default_runtime.yaml

task:
  type: ClassificationTask

model:
{MODEL_BLOCK}

train_dataloader:
  batch_size: 16
  num_workers: 4
  dataset:
    type: StandardClassificationDataset
    data_root: datasets/classfication/MIDOG
    ann_file: standardized/mido_cls_train.json
    pipeline:
      - type: LoadImageFromFile
      - type: Resize
        size: [224, 224]
      - type: Normalize
        mean: [123.675, 116.28, 103.53]
        std: [58.395, 57.12, 57.375]
      - type: PackInputs
        keys: ['img']

val_dataloader:
  batch_size: 16
  num_workers: 4
  dataset:
    type: StandardClassificationDataset
    data_root: datasets/classfication/MIDOG
    ann_file: standardized/mido_cls_val.json
    pipeline:
      - type: LoadImageFromFile
      - type: Resize
        size: [224, 224]
      - type: Normalize
        mean: [123.675, 116.28, 103.53]
        std: [58.395, 57.12, 57.375]
      - type: PackInputs
        keys: ['img']

val_evaluator:
  metrics:
    - type: Accuracy
    - type: F1_Score
    - type: Precision
    - type: Recall
  plotters:
    - type: ROCPlotter
    - type: ConfusionMatrixPlotter

optim_wrapper:
  optimizer:
    type: AdamW
    lr: 0.0001
    weight_decay: 0.05

runner:
  max_epochs: {EPOCHS}

default_hooks:
  checkpoint_hook:
    type: CheckpointHook
    interval: 10
    save_best: 'accuracy'
    rule: 'greater'

custom_hooks:
  plotter_hook:
    type: TrainingProgressPlotterHook

env:
  cudnn_benchmark: true
  device: cuda
"""

configs = [
    ('timm_resnet18_mido', "  type: TimmClassifier\n  architecture: resnet18\n  num_classes: 2\n  pretrained: true", 100),
    ('timm_resnet50_mido', "  type: TimmClassifier\n  architecture: resnet50\n  num_classes: 2\n  pretrained: true", 100),
    ('timm_efficientnet_b3_mido', "  type: TimmClassifier\n  architecture: efficientnet_b3\n  num_classes: 2\n  pretrained: true", 100),
    ('timm_mobilenetv3_mido', "  type: TimmClassifier\n  architecture: mobilenetv3_large_100\n  num_classes: 2\n  pretrained: true", 100),
    ('yolo_v8m_cls_mido', "  type: UltralyticsClsAdapter\n  yaml_model: yolov8m-cls.pt\n  num_classes: 2", 100),
    ('yolo_26m_cls_mido', "  type: UltralyticsClsAdapter\n  yaml_model: yolo26m-cls.pt\n  num_classes: 2", 100),
    ('timm_resnet50_mido_test', "  type: TimmClassifier\n  architecture: resnet50\n  num_classes: 2\n  pretrained: true", 1)
]

os.makedirs('configs/classify', exist_ok=True)

for name, model_block, epochs in configs:
    content = base_template.replace('{MODEL_BLOCK}', model_block).replace('{EPOCHS}', str(epochs))
    if name == 'timm_resnet50_mido_test':
        content = content.replace('batch_size: 16', 'batch_size: 4')
    with open(f"configs/classify/{name}.yaml", "w", encoding='utf-8') as f:
        f.write(content)

print("Restored ALL 7 configs successfully.")
