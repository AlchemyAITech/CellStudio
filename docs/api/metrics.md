# Metrics

CellStudio metrics follow a registry-based plugin pattern.
Each metric is a self-contained class that computes a single scalar
value from ground truth and predictions.

## BaseMetric

::: cellstudio.metrics.base.BaseMetric

## Metric Registry

```python
from cellstudio.metrics.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register('MyMetric')
class MyMetric(BaseMetric):
    def __init__(self, **kwargs):
        pass

    def compute(self, y_true, y_pred, y_prob=None, **kwargs):
        return float(...)
```

## Built-in Metrics

### Classification

| Metric | Registry Name | Description |
|---|---|---|
| Accuracy | `Accuracy` | Overall classification accuracy |
| Precision | `Precision` | Weighted precision |
| Recall | `Recall` | Weighted recall |
| F1Score | `F1Score` | Weighted F1 |
| AUC | `AUC` | Area under ROC curve |
| PRAUC | `PRAUC` | Area under PR curve |
| Kappa | `Kappa` | Cohen's kappa coefficient |
| ICC | `ICC` | Intraclass correlation coefficient |

### Detection

| Metric | Registry Name | Description |
|---|---|---|
| DetPrecision | `DetPrecision` | Detection precision at IoU threshold |
| DetRecall | `DetRecall` | Detection recall |
| DetF1 | `DetF1` | Detection F1 score |
| DetMAP50 | `DetMAP50` | Mean average precision @ IoU=0.50 |
| DetCountError | `DetCountError` | Absolute cell count error |

### Segmentation

| Metric | Registry Name | Description |
|---|---|---|
| SegDice | `SegDice` | Dice similarity coefficient |
| SegmIoU | `SegmIoU` | Mean intersection over union |
| SegPQ | `SegPQ` | Panoptic quality |
| SegAJI | `SegAJI` | Aggregated Jaccard index |
| SegHD95 | `SegHD95` | 95th percentile Hausdorff distance |
| SegAllMetrics | `SegAllMetrics` | Computes all segmentation metrics at once |

## MetricCollection

::: cellstudio.metrics.registry.MetricCollection
