import torch
from .base import Hook
from .registry import HOOK_REGISTRY

@HOOK_REGISTRY.register('AmpOptimizerHook')
class AmpOptimizerHook(Hook):
    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def after_train_iter(self, runner, **kwargs):
        runner.optimizer.zero_grad()
        loss = runner.outputs['loss']
        
        if runner.use_amp:
            runner.scaler.scale(loss).backward()
            if self.grad_clip:
                runner.scaler.unscale_(runner.optimizer)
                torch.nn.utils.clip_grad_norm_(runner.model.parameters(), self.grad_clip)
            runner.scaler.step(runner.optimizer)
            runner.scaler.update()
        else:
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(runner.model.parameters(), self.grad_clip)
            runner.optimizer.step()
