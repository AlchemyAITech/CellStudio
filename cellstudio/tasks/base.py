from abc import ABC, abstractmethod
from typing import Dict, Any
import torch

from ..env.dist_env import init_dist
from ..env.seed import set_random_seed
from ..engine.runner.epoch_runner import EpochBasedRunner
from ..engine.hooks.registry import HookRegistry
from ..models.builder import MODEL_REGISTRY
from ..evaluation.evaluator import Evaluator

class BaseTask(ABC):
    """
    The Zenith Task Commander.
    Takes a monolithic resolved Config and spawns the entire universe of Runner,
    Datasets, Adapters, Pipelines, and Evaluators automatically.
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.work_dir = cfg.get('work_dir', './runs')
        self.model = None
        self.runner = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.evaluator = None

    def build_env(self):
        env_cfg = self.cfg.get('env', {})
        init_dist(launcher=env_cfg.get('launcher', 'none'), backend=env_cfg.get('backend', 'nccl'))
        if 'seed' in env_cfg:
            set_random_seed(env_cfg['seed'], deterministic=env_cfg.get('deterministic', False))

    def build_model(self):
        model_cfg = self.cfg.get('model')
        if not model_cfg:
            raise ValueError("Config missing 'model' definition")
        self.model = MODEL_REGISTRY.build(model_cfg)
        
        device = self.cfg.get('env', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

    @abstractmethod
    def build_datasets(self):
        pass

    def build_evaluator(self):
        val_evaluator_cfg = self.cfg.get('val_evaluator', {})
        metrics = val_evaluator_cfg.get('metrics', [])
        plotters = val_evaluator_cfg.get('plotters', [])
        self.evaluator = Evaluator(metrics_cfg=metrics, plotters_cfg=plotters)

    def build_runner(self):
        optim_wrapper_cfg = self.cfg.get('optim_wrapper', {})
        optim_cfg = optim_wrapper_cfg.get('optimizer', {}) if optim_wrapper_cfg else self.cfg.get('optimizer', {})
        optimizer = None
        if optim_cfg and hasattr(torch.optim, optim_cfg.get('type', 'AdamW')):
            opt_cfg_copy = optim_cfg.copy()
            opt_type = opt_cfg_copy.pop('type')
            optimizer = getattr(torch.optim, opt_type)(self.model.parameters(), **opt_cfg_copy)

        runner_cfg = dict(self.cfg.get('runner', {'type': 'EpochBasedRunner', 'max_epochs': 100}))
        runner_type = runner_cfg.pop('type', 'EpochBasedRunner')
        runner_cfg.pop('val_interval', None)  # Handled in epoch_runner.train() via self.cfg
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
                **runner_cfg
            )
        else:
            raise NotImplementedError(f"Runner {runner_type} not supported.")

        if self.evaluator and self.val_dataloader:
            from ..engine.hooks.eval_hook import EvalHook
            self.runner.register_hook(EvalHook(evaluator=self.evaluator))

        hooks_cfg = self.cfg.get('default_hooks', {})
        for hook_name, hk_cfg in hooks_cfg.items():
            if hk_cfg:
                hk_instance = HookRegistry.build(hk_cfg)
                self.runner.register_hook(hk_instance)

    def execute(self, mode='train'):
        self.build_env()
        self.build_model()
        self.build_datasets()
        self.build_evaluator()
        self.build_runner()
        
        if mode == 'train':
            self.runner.train()
        elif mode == 'test':
            self.runner.val()
