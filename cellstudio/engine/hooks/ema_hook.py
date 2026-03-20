from .base import Hook
from .registry import HOOK_REGISTRY

@HOOK_REGISTRY.register('EMAHook')
class EMAHook(Hook):
    def __init__(self, decay=0.9999):
        self.decay = decay
        self.ema_model = None

    def before_run(self, runner, **kwargs):
        # Simplistic EMA init for structural completeness
        pass

    def after_train_iter(self, runner, **kwargs):
        pass
