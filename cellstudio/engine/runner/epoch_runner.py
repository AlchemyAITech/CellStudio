"""Epoch-based training runner.

Implements the standard epoch loop with hook-driven lifecycle events,
manual learning-rate scheduling, and optional AMP support.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from .base_runner import BaseRunner

logger = logging.getLogger(__name__)


class EpochBasedRunner(BaseRunner):
    """Concrete runner that iterates over data in epoch-major order.

    The training loop fires lifecycle hooks at well-defined points,
    allowing logging, checkpointing, evaluation, and custom logic to be
    injected without modifying the runner itself.

    Args:
        model: Neural network module.
        optimizer: PyTorch optimizer (may be ``None`` for eval-only runs).
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        max_epochs: Number of epochs to train.
        work_dir: Output directory for artifacts.
        meta: Optional metadata dictionary.
        use_amp: Enable automatic mixed precision.
        cfg: Full resolved configuration dictionary (used for LR
            scheduler params and validation interval).

    Lifecycle events (in order per epoch):
        1. ``before_run`` (once)
        2. ``before_train_epoch``
        3. ``before_train_iter`` / ``after_train_iter`` (per batch)
        4. ``after_train_epoch``
        5. ``before_val_epoch`` / ``before_val_iter`` / ``after_val_iter``
           / ``after_val_epoch`` (if validation is triggered)
        6. ``after_run`` (once)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        max_epochs: int = 100,
        work_dir: Optional[str] = None,
        meta: Optional[Dict] = None,
        use_amp: bool = False,
        cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(model, optimizer, work_dir, meta, use_amp)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_epochs = max_epochs
        self.cfg = cfg or {}

        if self.train_dataloader is not None:
            self.max_iters = self.max_epochs * len(self.train_dataloader)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Execute the full training loop over ``max_epochs`` epochs.

        At each epoch the runner iterates over the training dataloader,
        optionally adjusts learning rates via config-driven schedulers,
        and fires hook events.  Validation is run every
        ``runner.val_interval`` epochs (and always on the final epoch).
        """
        self.model.train()
        self.call_hook('before_run')

        # Capture base LRs for manual scheduler calculation
        base_lrs = (
            [group['lr'] for group in self.optimizer.param_groups]
            if self.optimizer
            else []
        )
        sched_cfg = self.cfg.get('param_scheduler', [])

        while self.epoch < self.max_epochs:
            self.call_hook('before_train_epoch')

            for i, data_batch in enumerate(self.train_dataloader):
                self.inner_iter = i

                # --- Manual LR scheduling ---------------------------------
                self._apply_lr_schedule(sched_cfg, base_lrs)

                self.call_hook(
                    'before_train_iter', batch_idx=i, data_batch=data_batch,
                )

                # --- Forward pass (with optional AMP) ---------------------
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.model.forward_train(
                        data_batch['imgs'],
                        data_batch.get('data_samples'),
                    )

                self.outputs = outputs
                self.call_hook(
                    'after_train_iter',
                    batch_idx=i,
                    data_batch=data_batch,
                    outputs=outputs,
                )
                self.iter += 1

            self.call_hook('after_train_epoch')

            # --- Conditional validation -----------------------------------
            val_interval = self.cfg.get('runner', {}).get('val_interval', 1)
            is_last = (self.epoch + 1) >= self.max_epochs
            if self.val_dataloader and (
                (self.epoch + 1) % val_interval == 0 or is_last
            ):
                self.val()

            self.epoch += 1

        self.call_hook('after_run')

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def val(self) -> None:
        """Run a single validation epoch over ``val_dataloader``.

        The method computes both validation loss (in train mode to avoid
        NMS in detection heads) and metric predictions (in eval mode).
        Results are communicated to hooks via the ``after_val_epoch``
        event with a ``metrics`` keyword argument.
        """
        self.call_hook('before_val_epoch')
        val_losses: list[float] = []

        for i, data_batch in enumerate(self.val_dataloader):
            self.data_batch = data_batch
            self.call_hook(
                'before_val_iter', batch_idx=i, data_batch=data_batch,
            )

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                # Compute val loss (train mode avoids NMS in detection heads)
                self.model.train()
                try:
                    loss_dict = self.model.forward_train(
                        data_batch['imgs'],
                        data_batch.get('data_samples'),
                    )
                    if 'loss' in loss_dict:
                        val_losses.append(loss_dict['loss'].item())
                except Exception:
                    logger.debug(
                        "Val loss computation failed for batch %d; skipping.",
                        i, exc_info=True,
                    )

                # Compute predictions for metric evaluation
                self.model.eval()
                outputs = self.model.forward_test(
                    data_batch['imgs'],
                    data_batch.get('data_samples'),
                )

            self.outputs = outputs
            self.call_hook(
                'after_val_iter',
                batch_idx=i,
                data_batch=data_batch,
                outputs=outputs,
            )

        metrics: Dict[str, float] = {}
        if val_losses:
            metrics['val_loss'] = sum(val_losses) / len(val_losses)

        self.call_hook('after_val_epoch', metrics=metrics)
        self.model.train()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_lr_schedule(
        self,
        sched_cfg: list,
        base_lrs: list[float],
    ) -> None:
        """Apply config-driven LR schedule to the optimizer in-place.

        Supports ``LinearLR`` warm-up and ``MultiStepLR`` decay.

        Args:
            sched_cfg: List of scheduler config dicts from the YAML.
            base_lrs: Initial learning rates captured at the start of
                training.
        """
        if not (self.optimizer and sched_cfg):
            return

        lr_factor = 1.0
        for cfg_s in sched_cfg:
            stype = cfg_s.get('type', '')
            if stype == 'LinearLR':
                begin = cfg_s.get('begin', 0)
                end = cfg_s.get('end', 500)
                start_factor = cfg_s.get('start_factor', 0.001)
                if self.iter <= end:
                    progress = (self.iter - begin) / max(1, end - begin)
                    progress = min(max(progress, 0.0), 1.0)
                    lr_factor *= start_factor + (1.0 - start_factor) * progress
            elif stype == 'MultiStepLR':
                milestones = cfg_s.get('milestones', [40, 80])
                gamma = cfg_s.get('gamma', 0.1)
                for m in milestones:
                    if self.epoch >= m:
                        lr_factor *= gamma

        for idx, group in enumerate(self.optimizer.param_groups):
            group['lr'] = base_lrs[idx] * lr_factor
