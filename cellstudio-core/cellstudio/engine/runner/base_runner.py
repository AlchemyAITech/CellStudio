"""Base runner providing the event-driven lifecycle framework.

This module defines :class:`BaseRunner`, the abstract foundation for all
training/validation loop implementations in CellStudio.  It manages a
hook system that decouples cross-cutting concerns (logging, checkpointing,
evaluation) from the iteration logic itself.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch

from ..hooks.base import Hook


class BaseRunner(ABC):
    """Abstract event-driven runner that separates loop control from business logic.

    Concrete subclasses (e.g. :class:`EpochBasedRunner`) implement the
    ``train()`` and ``val()`` methods while this base class provides
    hook management and shared state tracking.

    Args:
        model: The neural network module (already moved to the target device).
        optimizer: Optional optimizer.  When ``None`` the runner operates
            in inference-only mode.
        work_dir: Directory for saving checkpoints, logs, and artifacts.
        meta: Arbitrary metadata dictionary attached to the run.
        use_amp: Whether to enable automatic mixed-precision training.

    Attributes:
        epoch: Current epoch index (zero-based).
        iter: Global iteration counter across all epochs.
        inner_iter: Iteration counter within the current epoch.
        max_epochs: Total number of epochs the runner will execute.
        max_iters: Total number of iterations (``max_epochs * iters_per_epoch``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        work_dir: Optional[str] = None,
        meta: Optional[Dict] = None,
        use_amp: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.work_dir = work_dir
        self.meta = meta or {}
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self._hooks: List[Hook] = []

        # Core state tracking
        self.epoch: int = 0
        self.iter: int = 0
        self.inner_iter: int = 0
        self.max_epochs: int = 0
        self.max_iters: int = 0

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hook(self, hook: Hook) -> None:
        """Append a lifecycle hook to the runner's hook chain.

        Hooks are called in registration order at each lifecycle event
        (``before_run``, ``after_train_iter``, etc.).

        Args:
            hook: An instance of :class:`Hook` or any subclass.
        """
        self._hooks.append(hook)

    def call_hook(self, fn_name: str, *args, **kwargs) -> None:
        """Invoke the named method on every registered hook.

        Args:
            fn_name: Name of the lifecycle method to call (e.g.
                ``'before_train_epoch'``).
            *args: Positional arguments forwarded to each hook.
            **kwargs: Keyword arguments forwarded to each hook.
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self, *args, **kwargs)

    # ------------------------------------------------------------------
    # Abstract loop methods
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self) -> None:
        """Execute the full training loop."""

    @abstractmethod
    def val(self) -> None:
        """Execute a single validation epoch."""
