# Inference

The inference module provides a unified API for running predictions
with trained CellStudio models.

## CellStudioInferencer

::: cellstudio.inference.inferencer.CellStudioInferencer

### Quick Start

```python
from cellstudio.inference.inferencer import CellStudioInferencer

# Initialize with config + weights
infer = CellStudioInferencer(
    config_path='configs/classify/resnet50_mido.yaml',
    weight_path='runs/best_cls/best.pth',
    device='cuda',
)

# Run inference
result = infer('data/test_image.png')
print(f"Class: {result['class_id']}, Confidence: {result['confidence']:.3f}")
```

### Output Format

#### Classification

```python
{
    'class_id': 1,
    'confidence': 0.97,
    'probabilities': [0.03, 0.97],
}
```

## Evaluator

::: cellstudio.evaluation.evaluator.Evaluator
