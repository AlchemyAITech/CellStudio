from .base import BaseTask
from .classification import ClassificationTask
from .object_detection import ObjectDetectionTask
from .segmentation import InstanceSegmentationTask

__all__ = ['BaseTask', 'ClassificationTask', 'ObjectDetectionTask', 'InstanceSegmentationTask']
