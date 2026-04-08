"""Hook registry.

All lifecycle hooks (logger, checkpoint, optimizer, eval, etc.)
register themselves here so the task's ``build_runner`` method can
instantiate them from ``default_hooks`` configuration.
"""

from cellstudio.core.registry import Registry

HOOK_REGISTRY = Registry('hook')

# Backward-compatible alias used in tasks/base.py.
HookRegistry = HOOK_REGISTRY
