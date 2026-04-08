"""Abstract base class for all CellStudio metrics.

Every concrete metric (Accuracy, F1, mAP, Dice, etc.) must subclass
:class:`BaseMetric` and implement the :meth:`compute` method.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseMetric(ABC):
    """Interface that all metric implementations must satisfy.

    Subclasses are instantiated by the :class:`MetricRegistry` and
    called by the :class:`Evaluator` at the end of each validation
    epoch.

    Args:
        **kwargs: Metric-specific configuration (e.g. ``num_classes``,
            ``threshold``).
    """

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        ...

    @abstractmethod
    def compute(
        self,
        y_true: Any,
        y_pred: Any,
        y_prob: Optional[Any] = None,
        **kwargs,
    ) -> float:
        """Calculate the metric value.

        Args:
            y_true: Ground-truth labels or targets.
            y_pred: Model predictions.
            y_prob: Optional predicted probabilities (required by
                metrics like AUC and PR-AUC).
            **kwargs: Additional metric-specific arguments.

        Returns:
            Scalar metric value.
        """
        ...
