import time
from typing import Dict, Any, List

class BaseHook:
    """
    Base class for all Engine/Adapter execution Hooks.
    Subclasses should override the relevant events.
    """
    
    def on_train_begin(self, adapter: Any, **kwargs):
        pass

    def on_epoch_begin(self, adapter: Any, epoch: int, **kwargs):
        pass
        
    def on_train_step_end(self, adapter: Any, step: int, loss: float, **kwargs):
        pass

    def on_epoch_end(self, adapter: Any, epoch: int, metrics: Dict[str, float], **kwargs):
        pass

    def on_train_end(self, adapter: Any, **kwargs):
        pass


class RemoteProgressHook(BaseHook):
    """
    A concrete hook implementation intended to push telemetry to external services (like Redis/Celery).
    When the No-Code UI runs a task, this hook captures the loop progress and sends it back.
    """
    
    def __init__(self, job_id: str, redis_url: str = None):
        self.job_id = job_id
        self.redis_url = redis_url
        self.start_time = 0
        
    def _push_to_remote(self, payload: Dict):
        # Placeholder for actual Redis/HTTP push. 
        # In production, this would use redis.Redis.publish() or similar.
        # print(f"[Hook -> API] Job {self.job_id}: {payload}")
        pass

    def on_train_begin(self, adapter: Any, **kwargs):
        self.start_time = time.time()
        self._push_to_remote({"status": "RUNNING", "message": "Training started."})

    def on_epoch_end(self, adapter: Any, epoch: int, metrics: Dict[str, float], **kwargs):
        elapsed = time.time() - self.start_time
        payload = {
            "status": "RUNNING",
            "epoch": epoch,
            "metrics": metrics,
            "elapsed_seconds": elapsed
        }
        self._push_to_remote(payload)

    def on_train_end(self, adapter: Any, **kwargs):
        self._push_to_remote({"status": "COMPLETED", "message": "Training finished."})


class HookManager:
    """
    Manages a collection of hooks and broadcasts events to all of them.
    This is embedded within the Adapter/Trainer.
    """
    def __init__(self):
        self.hooks: List[BaseHook] = []
        
    def add_hook(self, hook: BaseHook):
        self.hooks.append(hook)
        
    def trigger(self, event_name: str, adapter: Any, **kwargs):
        for hook in self.hooks:
            method = getattr(hook, event_name, None)
            if method:
                method(adapter, **kwargs)
