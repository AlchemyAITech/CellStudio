# Data Structures

Core data structures that serve as the universal contract between
model adapters and the evaluation/visualization subsystems.

## InstanceData

::: cellstudio.structures.results.InstanceData

## DataSample

::: cellstudio.structures.results.DataSample

### Usage

```python
from cellstudio.structures.results import DataSample, InstanceData
import torch

sample = DataSample(
    img_path='/data/images/001.png',
    img_shape=(1024, 1024),
    ori_shape=(2048, 2048),
    gt_instances=InstanceData(
        bboxes=torch.tensor([[100, 200, 300, 400]]),
        labels=torch.tensor([0]),
    ),
)
```

## CellStudioInferResult

::: cellstudio.structures.results.CellStudioInferResult

Adapters must convert their framework-specific outputs into this
format so that evaluation code can operate without backend-specific
branching:

```python
from cellstudio.structures.results import CellStudioInferResult

result = CellStudioInferResult(
    bboxes=torch.tensor([[10, 20, 100, 200]]),
    labels=torch.tensor([0]),
    scores=torch.tensor([0.95]),
)
```
