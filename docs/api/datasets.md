# Datasets

CellStudio datasets parse annotation files into standardized
dictionaries and apply configurable transform pipelines.

## BaseDataset

::: cellstudio.datasets.base.BaseDataset

### Data Item Format

Every dataset item is a dictionary with at least:

```python
{
    'img_path': '/path/to/image.png',
    'img_id': 42,
    'img_shape': (1024, 1024),
    'gt_bboxes': np.array([[x1, y1, x2, y2], ...]),  # (N, 4) float32
    'gt_labels': np.array([0, 1, ...]),                # (N,) int64
}
```

## Concrete Datasets

### MIDODataset

Parses the custom MIDO JSON format with `classes` and `items` arrays.
Automatically resolves absolute paths from legacy annotation files.

### TileMIDODataset

Extends MIDO loading with large-image tiling. Extracts fixed-size
patches and filters by minimum annotation count.

### CellposeSegmentationDataset

Loads instance segmentation annotations with per-cell binary masks
for Cellpose and UNet training.

## Dataset Registry

```python
from cellstudio.datasets.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register('MyDataset')
class MyDataset(BaseDataset):
    def _load_data_list(self):
        ...
```
