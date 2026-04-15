"""Pipeline transform registry.

All data-processing transforms (loading, augmentation, formatting)
register themselves here so the ``Compose`` pipeline can resolve them
from configuration dictionaries.
"""

from ..core.registry import Registry

PIPELINE_REGISTRY = Registry('pipeline')
