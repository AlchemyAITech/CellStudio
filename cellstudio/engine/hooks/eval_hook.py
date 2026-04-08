from .base import Hook
from .registry import HOOK_REGISTRY

@HOOK_REGISTRY.register('EvalHook')
class EvalHook(Hook):
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def after_val_iter(self, runner, **kwargs):
        self.evaluator.process(runner.data_batch, runner.outputs)

    def after_val_epoch(self, runner, **kwargs):
        eval_metrics = self.evaluator.evaluate(runner.work_dir)
        # Merge dicts such as val_loss generated purely by the runner forwards
        incoming_metrics = kwargs.get('metrics', {})
        if incoming_metrics:
            eval_metrics.update(incoming_metrics)
            
        runner.val_metrics = eval_metrics
