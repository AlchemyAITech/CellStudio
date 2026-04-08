"""Base hook defining the runner lifecycle interface.

All custom hooks should subclass :class:`Hook` and override only the
lifecycle methods they need.  The default implementations are no-ops.
"""


class Hook:
    """Base class for all runner lifecycle hooks.

    Hooks are called by the runner at well-defined points during training
    and validation.  Override any subset of the methods below to inject
    custom behavior (logging, checkpointing, evaluation, etc.).

    The ``runner`` argument passed to every method is the active
    :class:`~cellstudio.engine.runner.base_runner.BaseRunner` instance,
    giving hooks full read/write access to model state, optimizer, and
    iteration counters.
    """

    # --- Run-level events -------------------------------------------------

    def before_run(self, runner, **kwargs) -> None:
        """Called once before the first training epoch begins."""

    def after_run(self, runner, **kwargs) -> None:
        """Called once after the last training epoch completes."""

    # --- Training epoch events -------------------------------------------

    def before_train_epoch(self, runner, **kwargs) -> None:
        """Called at the start of each training epoch."""

    def after_train_epoch(self, runner, **kwargs) -> None:
        """Called at the end of each training epoch."""

    # --- Training iteration events ----------------------------------------

    def before_train_iter(self, runner, **kwargs) -> None:
        """Called before each training iteration (forward pass).

        Keyword Args:
            batch_idx (int): Index of the current batch within the epoch.
            data_batch (dict): The current mini-batch dictionary.
        """

    def after_train_iter(self, runner, **kwargs) -> None:
        """Called after each training iteration (after loss.backward).

        Keyword Args:
            batch_idx (int): Index of the current batch within the epoch.
            data_batch (dict): The current mini-batch dictionary.
            outputs (dict): Model output dictionary containing ``'loss'``.
        """

    # --- Validation epoch events ------------------------------------------

    def before_val_epoch(self, runner, **kwargs) -> None:
        """Called at the start of each validation epoch."""

    def after_val_epoch(self, runner, **kwargs) -> None:
        """Called at the end of each validation epoch.

        Keyword Args:
            metrics (dict): Scalar metrics computed during validation
                (e.g. ``{'val_loss': 0.42}``).
        """

    # --- Validation iteration events --------------------------------------

    def before_val_iter(self, runner, **kwargs) -> None:
        """Called before each validation iteration.

        Keyword Args:
            batch_idx (int): Index of the current batch.
            data_batch (dict): The current mini-batch dictionary.
        """

    def after_val_iter(self, runner, **kwargs) -> None:
        """Called after each validation iteration.

        Keyword Args:
            batch_idx (int): Index of the current batch.
            data_batch (dict): The current mini-batch dictionary.
            outputs: Model prediction outputs.
        """
