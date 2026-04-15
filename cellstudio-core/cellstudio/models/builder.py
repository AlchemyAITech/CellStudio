"""Model component registry.

Provides a global ``MODEL_REGISTRY`` instance that model adapters (YOLO,
Cellpose, UNet, Timm, etc.) register themselves into via the
``@MODEL_REGISTRY.register()`` decorator.
"""

from ..core.registry import Registry

MODEL_REGISTRY = Registry('model')


def build_model(cfg: dict):
    """Convenience wrapper around ``MODEL_REGISTRY.build``.

    Args:
        cfg: A dictionary with at least a ``'type'`` key identifying
            the registered model adapter class.

    Returns:
        An instantiated model adapter.
    """
    return MODEL_REGISTRY.build(cfg)
