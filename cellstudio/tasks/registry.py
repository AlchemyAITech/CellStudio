class TaskRegistry:
    """Registry pattern insulating high-level tasks."""
    _registry = {}

    @classmethod
    def register(cls, name=None):
        def inner_wrapper(wrapped_class):
            plugin_name = name if name is not None else wrapped_class.__name__
            if plugin_name in cls._registry:
                raise ValueError(f"Task '{plugin_name}' already registered.")
            cls._registry[plugin_name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def build(cls, cfg: dict):
        if cfg is None:
            return None
        cfg_copy = cfg.copy()
        if 'type' not in cfg_copy:
            raise KeyError("Task config must contain a 'type' key.")
        plugin_type = cfg_copy.pop('type')
        if plugin_type not in cls._registry:
            raise KeyError(f"Task '{plugin_type}' is unknown.")
        return cls._registry[plugin_type](**cfg_copy)

TASK_REGISTRY = TaskRegistry()
