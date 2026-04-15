"""Abstract task commander.

:class:`BaseTask` orchestrates the full training/evaluation lifecycle
by composing a Runner, Datasets, Model, Evaluator, and Hooks from a
single resolved configuration dictionary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch

from ..engine.hooks.registry import HookRegistry
from ..engine.runner.epoch_runner import EpochBasedRunner
from ..env.dist_env import init_dist
from ..env.seed import set_random_seed
from ..evaluation.evaluator import Evaluator
from ..models.builder import MODEL_REGISTRY


class BaseTask(ABC):
    """Abstract base for all CellStudio task types.

    A task is the top-level entry point that takes a monolithic resolved
    config and spawns the entire universe of Runner, Datasets, Adapters,
    Pipelines, and Evaluators automatically.

    Subclasses must implement :meth:`build_datasets` to wire up the
    task-specific data loading logic.

    Args:
        cfg: Fully resolved configuration dictionary (from
            :meth:`Config.fromfile`).

    Attributes:
        work_dir: Output directory for checkpoints, logs, and plots.
        model: The constructed model adapter (populated by
            :meth:`build_model`).
        runner: The training runner (populated by :meth:`build_runner`).
        evaluator: The validation evaluator (populated by
            :meth:`build_evaluator`).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.work_dir: str = cfg.get('work_dir', './runs')
        self.model = None
        self.runner = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.evaluator = None

    # ------------------------------------------------------------------
    # Build stages
    # ------------------------------------------------------------------

    def build_env(self) -> None:
        """Initialize distributed environment and random seeds.

        Reads ``env.launcher``, ``env.backend``, ``env.seed``, and
        ``env.deterministic`` from the config.
        """
        env_cfg = self.cfg.get('env', {})
        init_dist(
            launcher=env_cfg.get('launcher', 'none'),
            backend=env_cfg.get('backend', 'nccl'),
        )
        if 'seed' in env_cfg:
            set_random_seed(
                env_cfg['seed'],
                deterministic=env_cfg.get('deterministic', False),
            )

    def build_model(self) -> None:
        """Instantiate and move the model to the configured device.

        Raises:
            ValueError: If the config is missing a ``'model'`` section.
        """
        model_cfg = self.cfg.get('model')
        if not model_cfg:
            raise ValueError("Config missing 'model' definition.")
        self.model = MODEL_REGISTRY.build(model_cfg)

        device = self.cfg.get('env', {}).get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu',
        )
        self.model.to(device)

    @abstractmethod
    def build_datasets(self) -> None:
        """Build train and validation dataloaders.

        Must populate ``self.train_dataloader`` and
        ``self.val_dataloader``.
        """

    def build_evaluator(self) -> None:
        """Build the validation evaluator from config.

        Reads ``val_evaluator.metrics`` and ``val_evaluator.plotters``
        to construct an :class:`Evaluator` instance.
        """
        val_evaluator_cfg = self.cfg.get('val_evaluator', {})
        metrics = val_evaluator_cfg.get('metrics', [])
        plotters = val_evaluator_cfg.get('plotters', [])
        self.evaluator = Evaluator(metrics_cfg=metrics, plotters_cfg=plotters)

    def build_runner(self) -> None:
        """Assemble the EpochBasedRunner, optimizer, and hook chain.

        The optimizer is built from ``optim_wrapper.optimizer`` config.
        Hooks are instantiated from ``default_hooks`` and registered in
        declaration order.
        """
        # --- Optimizer ---------------------------------------------------
        optim_wrapper_cfg = self.cfg.get('optim_wrapper', {})
        optim_cfg = (
            optim_wrapper_cfg.get('optimizer', {})
            if optim_wrapper_cfg
            else self.cfg.get('optimizer', {})
        )
        optimizer = None
        if optim_cfg and hasattr(torch.optim, optim_cfg.get('type', 'AdamW')):
            opt_cfg_copy = optim_cfg.copy()
            opt_type = opt_cfg_copy.pop('type')
            optimizer = getattr(torch.optim, opt_type)(
                self.model.parameters(), **opt_cfg_copy,
            )

        # --- Runner ------------------------------------------------------
        runner_cfg = dict(
            self.cfg.get('runner', {'type': 'EpochBasedRunner', 'max_epochs': 100}),
        )
        runner_type = runner_cfg.pop('type', 'EpochBasedRunner')
        runner_cfg.pop('val_interval', None)  # Handled inside epoch_runner
        use_amp = self.cfg.get('use_amp', False)

        if runner_type == 'EpochBasedRunner':
            self.runner = EpochBasedRunner(
                model=self.model,
                optimizer=optimizer,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                work_dir=self.work_dir,
                use_amp=use_amp,
                cfg=self.cfg,
                **runner_cfg,
            )
        else:
            raise NotImplementedError(f"Runner '{runner_type}' not supported.")

        # --- Eval hook ---------------------------------------------------
        if self.evaluator and self.val_dataloader:
            from ..engine.hooks.eval_hook import EvalHook
            self.runner.register_hook(EvalHook(evaluator=self.evaluator))

        # --- Default hooks from config -----------------------------------
        hooks_cfg = self.cfg.get('default_hooks', {})
        for hook_name, hk_cfg in hooks_cfg.items():
            if hk_cfg:
                hk_instance = HookRegistry.build(hk_cfg)
                self.runner.register_hook(hk_instance)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, mode: str = 'train') -> None:
        """Build all components and execute the specified mode.

        Args:
            mode: Either ``'train'`` to run full training or ``'test'``
                to run evaluation only.
        """
        self.build_env()
        self.build_model()
        self.build_datasets()
        self.build_evaluator()
        self.build_runner()

        if mode == 'train':
            self.runner.train()
        elif mode == 'test':
            self.runner.val()
