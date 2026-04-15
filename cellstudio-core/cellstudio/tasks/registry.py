"""Task registry.

All concrete task classes (Classification, ObjectDetection,
InstanceSegmentation) register themselves here so that they can be
instantiated from YAML configuration via ``TASK_REGISTRY.build(cfg)``.
"""

from ..core.registry import Registry

TASK_REGISTRY = Registry('task')
