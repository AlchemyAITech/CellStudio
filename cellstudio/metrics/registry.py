"""Metric registry and collection utilities.

Provides ``METRIC_REGISTRY`` for registering individual metric classes
and ``MetricCollection`` for batch-computing a set of named metrics.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from ..core.registry import Registry
from .base import BaseMetric

logger = logging.getLogger(__name__)

METRIC_REGISTRY = Registry('metric')

# Backward-compatible alias used extensively in existing code.
MetricRegistry = METRIC_REGISTRY


class MetricCollection:
    """Convenience container that evaluates multiple named metrics at once.

    Args:
        metric_names: List of registered metric names to instantiate.
        **kwargs: Extra keyword arguments forwarded to each metric's
            constructor.

    Example:
        >>> coll = MetricCollection(['Accuracy', 'F1Score'])
        >>> results = coll.compute_all(y_true, y_pred)
    """

    def __init__(self, metric_names: list[str], **kwargs) -> None:
        self.metrics: Dict[str, BaseMetric] = {
            n: METRIC_REGISTRY.get(n)(**kwargs) for n in metric_names
        }

    def compute_all(
        self,
        y_true,
        y_pred,
        y_prob=None,
        **kwargs,
    ) -> Dict[str, float]:
        """Compute every registered metric and return a flat dict.

        Args:
            y_true: Ground-truth labels / targets.
            y_pred: Predicted labels / values.
            y_prob: Optional predicted probabilities.
            **kwargs: Forwarded to each metric's ``compute()`` method.

        Returns:
            Mapping from metric name to its scalar result.  Metrics
            that raise are logged and recorded as ``0.0``.
        """
        results: Dict[str, float] = {}
        for name, metric in self.metrics.items():
            try:
                results[name] = metric.compute(
                    y_true, y_pred, y_prob, **kwargs,
                )
            except Exception:
                logger.warning("Metric '%s' failed; recording 0.0.", name, exc_info=True)
                results[name] = 0.0
        return results
