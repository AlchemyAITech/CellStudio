"""YAML configuration engine with hierarchical inheritance.

Provides :class:`Config` which loads OmegaConf-based YAML files,
resolves ``_base_`` inheritance chains, and eagerly interpolates
variable references (e.g. ``${model.backbone}``).
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


class Config:
    """Recursive config loader with ``_base_`` inheritance support.

    This is modeled after the MMEngine / OpenMMLab config system but
    uses OmegaConf as the underlying merge engine.

    Example:
        >>> cfg = Config.fromfile('configs/detect/yolo_v8m_det_mido.yaml')
        >>> print(cfg.model.type)
        'UltralyticsDetAdapter'
    """

    @staticmethod
    def _merge_base(cfg_path: str) -> DictConfig:
        """Recursively load and merge ``_base_`` config files.

        Args:
            cfg_path: Absolute or relative path to a YAML config file.

        Returns:
            A fully merged :class:`DictConfig` with all ``_base_``
            ancestors resolved.

        Raises:
            FileNotFoundError: If *cfg_path* does not exist on disk.
        """
        cfg_path = Path(cfg_path).absolute()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        cfg = OmegaConf.load(cfg_path)

        if '_base_' in cfg:
            bases = cfg['_base_']
            if isinstance(bases, str):
                bases = [bases]

            merged_base = OmegaConf.create()
            for base_file in bases:
                base_path = cfg_path.parent / base_file
                base_cfg = Config._merge_base(str(base_path))
                merged_base = OmegaConf.merge(merged_base, base_cfg)

            del cfg['_base_']
            cfg = OmegaConf.merge(merged_base, cfg)

        return cfg

    @staticmethod
    def fromfile(cfg_path: str) -> DictConfig:
        """Load a config file, resolving inheritance and interpolations.

        This is the primary public entry point for configuration loading
        throughout CellStudio.

        Args:
            cfg_path: Path to the root YAML configuration file.

        Returns:
            A fully resolved, read-ready :class:`DictConfig`.
        """
        cfg = Config._merge_base(cfg_path)
        OmegaConf.resolve(cfg)
        return cfg
