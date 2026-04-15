import torch

from .base import Hook
from .registry import HOOK_REGISTRY


@HOOK_REGISTRY.register('AmpOptimizerHook')
class AmpOptimizerHook(Hook):
    def __init__(self, grad_clip=None):
        # OmegaConf DictConfig does not pass isinstance(dict); convert it
        if grad_clip is not None and hasattr(grad_clip, 'items'):
            self.grad_clip = dict(grad_clip)
        else:
            self.grad_clip = grad_clip

    def after_train_iter(self, runner, **kwargs):
        runner.optimizer.zero_grad()
        loss = runner.outputs['loss']
        
        if runner.use_amp:
            runner.scaler.scale(loss).backward()
            if self.grad_clip:
                runner.scaler.unscale_(runner.optimizer)
                if isinstance(self.grad_clip, dict):
                    torch.nn.utils.clip_grad_norm_(runner.model.parameters(), **self.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(runner.model.parameters(), max_norm=self.grad_clip)
            runner.scaler.step(runner.optimizer)
            runner.scaler.update()
        else:
            loss.backward()
            if self.grad_clip:
                if isinstance(self.grad_clip, dict):
                    torch.nn.utils.clip_grad_norm_(runner.model.parameters(), **self.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(runner.model.parameters(), max_norm=self.grad_clip)
            runner.optimizer.step()
