class Registry:
    def __init__(self, name):
        self.name = name
        self._registry = {}

    def register(self, module_name=None):
        def inner_wrapper(wrapped_class):
            name = module_name if module_name is not None else wrapped_class.__name__
            if name in self._registry:
                raise ValueError(f"Module '{name}' already registered in {self.name}.")
            self._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    def build(self, cfg: dict):
        if cfg is None:
            return None
        cfg_copy = cfg.copy()
        if 'type' not in cfg_copy:
            raise KeyError(f"Config for {self.name} must contain 'type' key.")
        plugin_type = cfg_copy.pop('type')
        if plugin_type not in self._registry:
            raise KeyError(f"Module '{plugin_type}' is unknown in {self.name}.")
        return self._registry[plugin_type](**cfg_copy)

HOOK_REGISTRY = Registry('hook')
HookRegistry = HOOK_REGISTRY
