"""Evaluation broker for decoupled metric computation and plotting.

The :class:`Evaluator` accumulates ground-truth / prediction pairs
across validation batches and computes all registered metrics at the
end of each validation epoch.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import torch

from ..metrics.registry import METRIC_REGISTRY
from ..plotting.registry import PLOTTER_REGISTRY

logger = logging.getLogger(__name__)


class Evaluator:
    """Orchestrates metric calculation and result persistence.

    The evaluator is wired into the validation loop via
    :class:`EvalHook`.  During validation, :meth:`process` is called
    once per batch.  After all batches, :meth:`evaluate` computes every
    registered metric and persists predictions to disk.

    Args:
        metrics_cfg: List of metric config dicts, each with a
            ``'type'`` key matching a registered metric name.
        plotters_cfg: List of plotter config dicts (currently deferred
            to post-hoc scripts to avoid matplotlib overhead in the
            training loop).

    Example:
        >>> evaluator = Evaluator(
        ...     metrics_cfg=[{'type': 'Accuracy'}, {'type': 'F1Score'}],
        ... )
    """

    def __init__(
        self,
        metrics_cfg: Optional[List[Dict]] = None,
        plotters_cfg: Optional[List[Dict]] = None,
    ) -> None:
        metrics_cfg = metrics_cfg or []
        plotters_cfg = plotters_cfg or []

        self.metrics = [METRIC_REGISTRY.build(cfg) for cfg in metrics_cfg]
        self.plotters = [PLOTTER_REGISTRY.build(cfg) for cfg in plotters_cfg]
        self._predictions: List[Any] = []
        self._data_samples: List[Dict] = []

    def process(self, data_batch: Dict, outputs: Any) -> None:
        """Accumulate a single batch of ground truth and predictions.

        Called by :class:`EvalHook` in ``after_val_iter``.

        Args:
            data_batch: The mini-batch dictionary from the dataloader.
            outputs: Model outputs for this batch.
        """
        self._data_samples.append(data_batch)
        self._predictions.append(outputs)

    def evaluate(self, work_dir: str) -> Dict[str, float]:
        """Compute all registered metrics and persist predictions.

        Args:
            work_dir: Directory to save ``predictions.pkl``.

        Returns:
            Flat dictionary mapping metric names to scalar values.
        """
        metrics_result: Dict[str, float] = {}

        # --- Unpack accumulated predictions --------------------------------
        y_true, y_pred, y_prob = self._unpack_predictions()

        # --- Compute metrics -----------------------------------------------
        for metric in self.metrics:
            try:
                metric_val = metric.compute(
                    y_true=y_true, y_pred=y_pred, y_prob=y_prob,
                )
                if isinstance(metric_val, dict):
                    metrics_result.update(metric_val)
                elif metric_val is not None:
                    metrics_result[metric.__class__.__name__] = metric_val
            except Exception:
                logger.warning(
                    "Metric '%s' failed during evaluation.",
                    metric.__class__.__name__,
                    exc_info=True,
                )

        # --- Persist predictions to disk -----------------------------------
        self._save_predictions(work_dir, y_true, y_pred, y_prob)

        # --- Cleanup -------------------------------------------------------
        self._predictions.clear()
        self._data_samples.clear()

        return metrics_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unpack_predictions(self):
        """Convert accumulated batch outputs into arrays for metrics.

        Returns:
            Tuple of ``(y_true, y_pred, y_prob)``.
        """
        y_prob = None

        # Classification format: dicts with 'gt_labels', 'preds', 'probs'
        if (
            self._predictions
            and isinstance(self._predictions[0], dict)
            and 'gt_labels' in self._predictions[0]
        ):
            y_true = torch.cat(
                [p['gt_labels'] for p in self._predictions],
            ).numpy()
            y_pred = torch.cat(
                [p['preds'] for p in self._predictions],
            ).numpy()
            y_prob = torch.cat(
                [p['probs'] for p in self._predictions],
            ).numpy()
        else:
            # Detection / segmentation: pass raw samples through
            y_true = [
                batch.get('data_samples', [])
                for batch in self._data_samples
            ]
            y_pred = self._predictions

        return y_true, y_pred, y_prob

    def _save_predictions(
        self,
        work_dir: str,
        y_true: Any,
        y_pred: Any,
        y_prob: Any,
    ) -> None:
        """Serialize predictions to a pickle file.

        Skips saving for instance segmentation results (which contain
        dense masks and would produce very large files).

        Args:
            work_dir: Output directory.
            y_true: Ground-truth data.
            y_pred: Prediction data.
            y_prob: Optional probability data.
        """
        if not self._predictions:
            return
        if isinstance(self._predictions[0], list):
            return  # Skip list-of-results (instance seg)

        # Skip known heavy types
        first_type_name = type(self._predictions[0]).__name__
        if 'CellStudioInferResult' in first_type_name:
            return

        try:
            os.makedirs(work_dir, exist_ok=True)
            pred_file = os.path.join(work_dir, 'predictions.pkl')
            with open(pred_file, 'wb') as f:
                pickle.dump(
                    {'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob},
                    f,
                )
        except Exception:
            logger.warning(
                "Failed to save predictions to %s.", work_dir, exc_info=True,
            )
