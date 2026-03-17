import pytest
from pathstudio.engine.hooks import HookManager, BaseHook

class DummyHook(BaseHook):
    def __init__(self):
        super().__init__()
        self.called_train_begin = False
        self.epoch_metrics = None
        
    def on_train_begin(self, adapter, **kwargs):
        self.called_train_begin = True
        
    def on_epoch_end(self, adapter, epoch, metrics, **kwargs):
        self.epoch_metrics = metrics

def test_hook_manager_subscription():
    manager = HookManager()
    hook = DummyHook()
    manager.add_hook(hook)
    
    # Test simple event activation
    manager.trigger("on_train_begin", adapter=None)
    assert hook.called_train_begin is True

def test_hook_manager_payload_delivery():
    manager = HookManager()
    hook = DummyHook()
    manager.add_hook(hook)
    
    # Test payload parsing inside triggers
    mock_metrics = {"loss": 0.45, "acc": 0.95}
    manager.trigger("on_epoch_end", adapter=None, epoch=1, metrics=mock_metrics)
    
    assert hook.epoch_metrics is not None
    assert hook.epoch_metrics["loss"] == 0.45
