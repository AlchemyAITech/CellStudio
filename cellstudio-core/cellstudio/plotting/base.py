"""Abstract base class for all CellStudio plotters.

Every concrete plotter (ROC curve, confusion matrix, etc.) must
subclass :class:`BasePlotter` and implement the :meth:`plot` method.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BasePlotter(ABC):
    """Interface that all visualization implementations must satisfy.

    Subclasses are instantiated by the :class:`PlotterRegistry` and
    called by the :class:`Evaluator` or by post-hoc analysis scripts.

    Args:
        **kwargs: Plotter-specific configuration.
    """

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        ...

    @abstractmethod
    def plot(
        self,
        save_dir: str,
        y_true: Optional[Any] = None,
        y_pred: Optional[Any] = None,
        y_prob: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """Generate the visualization and save it to disk.

        Args:
            save_dir: Directory where the output figure will be saved.
            y_true: Ground-truth labels or targets.
            y_pred: Model predictions.
            y_prob: Optional predicted probabilities.
            **kwargs: Additional plotter-specific arguments.
        """
        ...
