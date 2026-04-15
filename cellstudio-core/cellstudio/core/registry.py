"""Universal registry pattern for CellStudio component management.

This module provides a single, reusable ``Registry`` class that replaces
the previously duplicated registry implementations across models, tasks,
datasets, pipelines, metrics, plotters, and hooks.

Example:
    >>> from cellstudio.core import Registry
    >>> BACKBONE_REGISTRY = Registry('backbone')
    >>> @BACKBONE_REGISTRY.register('ResNet50')
    ... class ResNet50:
    ...     pass
    >>> model = BACKBONE_REGISTRY.build({'type': 'ResNet50'})
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)


class Registry:
    """A universal registry that maps string names to component classes.

    The registry enables a decorator-based registration pattern so that
    downstream modules can declare new components without modifying any
    central lookup table.  Each ``Registry`` instance maintains its own
    independent namespace.

    Args:
        name: Human-readable name for this registry (used in error messages).
        scope: Optional namespace prefix (default ``'cellstudio'``).

    Attributes:
        name: The human-readable registry name.
        scope: Namespace scope string.

    Example:
        >>> MODEL_REGISTRY = Registry('model')
        >>> @MODEL_REGISTRY.register('MyModel')
        ... class MyModel:
        ...     def __init__(self, depth=50): ...
        >>> instance = MODEL_REGISTRY.build({'type': 'MyModel', 'depth': 101})
    """

    def __init__(self, name: str, scope: str = 'cellstudio') -> None:
        self.name = name
        self.scope = scope
        self._registry: Dict[str, Type] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: Optional[str] = None) -> Callable:
        """Register a class under a given name (decorator factory).

        Args:
            name: Key to register the class with.  When ``None`` the
                class's ``__name__`` attribute is used instead.

        Returns:
            A decorator that registers the wrapped class and returns it
            unchanged.

        Raises:
            ValueError: If a class with the same name is already
                registered in this registry.
        """

        def inner_wrapper(cls: Type) -> Type:
            plugin_name = name if name is not None else cls.__name__
            if plugin_name in self._registry:
                raise ValueError(
                    f"[{self.scope}/{self.name}] "
                    f"'{plugin_name}' is already registered."
                )
            self._registry[plugin_name] = cls
            logger.debug(
                "[%s/%s] Registered '%s' -> %s",
                self.scope, self.name, plugin_name, cls.__qualname__,
            )
            return cls

        return inner_wrapper

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Type:
        """Retrieve a registered class by name.

        Args:
            name: The registered key.

        Returns:
            The class that was registered under *name*.

        Raises:
            KeyError: If *name* has not been registered.
        """
        if name not in self._registry:
            available = ', '.join(sorted(self._registry.keys())) or '(none)'
            raise KeyError(
                f"[{self.scope}/{self.name}] '{name}' not found. "
                f"Available: {available}"
            )
        return self._registry[name]

    # ------------------------------------------------------------------
    # Build from config dict
    # ------------------------------------------------------------------

    def build(self, cfg: Optional[Dict[str, Any]]) -> Any:
        """Instantiate a registered component from a config dictionary.

        The dictionary **must** contain a ``'type'`` key whose value
        matches a registered name.  All remaining key-value pairs are
        forwarded as keyword arguments to the class constructor.

        Args:
            cfg: Configuration dictionary.  ``None`` is a valid input
                and will return ``None``.

        Returns:
            An instance of the registered class, or ``None`` when
            *cfg* is ``None``.

        Raises:
            KeyError: If ``'type'`` is missing from *cfg* or if the
                type string has not been registered.
        """
        if cfg is None:
            return None

        cfg_copy = dict(cfg)  # Shallow copy; works with both dict and DictConfig
        if 'type' not in cfg_copy:
            raise KeyError(
                f"[{self.scope}/{self.name}] Config must contain a 'type' key."
            )
        plugin_type = cfg_copy.pop('type')
        cls = self.get(plugin_type)
        return cls(**cfg_copy)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def list_registered(self) -> list[str]:
        """Return a sorted list of all registered component names."""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return (
            f"Registry(name='{self.name}', scope='{self.scope}', "
            f"items={self.list_registered()})"
        )
