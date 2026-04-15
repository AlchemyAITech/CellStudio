"""Dataset registry.

Dataset classes (``MIDODataset``, ``TileMIDODataset``, etc.) are
registered here so that task modules can resolve them by string name
from configuration files.
"""

from ..core.registry import Registry

DATASET_REGISTRY = Registry('dataset')

# Backward-compatible alias used in some older scripts.
DatasetRegistry = DATASET_REGISTRY
