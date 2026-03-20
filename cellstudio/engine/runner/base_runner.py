import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch

from ..hooks.base import Hook

class BaseRunner(ABC):
    """
    The Zenith Event-Driven Runner Foundation.
    Separates the execution loop from the business logic.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 work_dir: str = None,
                 meta: Dict = None,
                 use_amp: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.work_dir = work_dir
        self.meta = meta or {}
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self._hooks: List[Hook] = []
        
        # Core State tracking variables
        self.epoch = 0
        self.iter = 0
        self.inner_iter = 0
        self.max_epochs = 0
        self.max_iters = 0
        
    def register_hook(self, hook: Hook):
        """Register a new lifecycle hook."""
        self._hooks.append(hook)
        
    def call_hook(self, fn_name: str, *args, **kwargs):
        """Trigger all registered hooks at the specified lifecycle event."""
        for hook in self._hooks:
            getattr(hook, fn_name)(self, *args, **kwargs)
            
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def val(self):
        pass
