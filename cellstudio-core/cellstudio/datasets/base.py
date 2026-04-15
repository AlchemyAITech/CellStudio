"""Abstract base dataset with pipeline integration.

Defines :class:`BaseDataset`, the foundation for all CellStudio
datasets.  Subclasses implement :meth:`_load_data_list` to parse
annotation files into a standardized list-of-dicts format.
"""

import copy
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from ..pipeline.compose import Compose


class BaseDataset(Dataset, ABC):
    """Abstract dataset that enforces dict-based data loading and pipeline transforms.

    Each dataset item is a dictionary containing at minimum an
    ``'img_path'`` key.  The ``pipeline`` (a :class:`Compose` instance)
    is applied to each item during ``__getitem__``.

    Args:
        data_root: Root directory containing the dataset files.
        ann_file: Annotation file path, relative to *data_root*.
        data_prefix: Mapping of data type to subdirectory prefix
            (e.g. ``{'img_path': 'images/'}``).
        pipeline: List of transform config dicts to build a
            :class:`Compose` pipeline.  ``None`` disables transforms.
        test_mode: When ``True``, the dataset operates in evaluation
            mode (may affect augmentation behavior).

    Attributes:
        data_list: Eagerly loaded list of data-info dictionaries.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        data_prefix: Optional[Dict[str, str]] = None,
        pipeline: Optional[List[Dict]] = None,
        test_mode: bool = False,
    ) -> None:
        self.data_root = data_root
        self.ann_file = (
            os.path.join(data_root, ann_file) if ann_file else data_root
        )
        self.data_prefix = data_prefix or {'img_path': ''}
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline) if pipeline else None

        # Eagerly load the dataset index
        self.data_list: List[Dict[str, Any]] = self._load_data_list()

    @abstractmethod
    def _load_data_list(self) -> List[Dict[str, Any]]:
        """Parse annotations into a list of standardized dictionaries.

        Each dictionary **must** contain at least an ``'img_path'`` key
        pointing to the image file on disk.

        Returns:
            List of data-info dictionaries.
        """

    def __len__(self) -> int:
        return len(self.data_list)

    def prepare_data(self, idx: int) -> Optional[Dict[str, Any]]:
        """Deep-copy and transform a single data item.

        Args:
            idx: Index into :attr:`data_list`.

        Returns:
            Transformed data dictionary, or ``None`` if the pipeline
            deliberately discards the item (e.g. empty tiles).
        """
        data_info = copy.deepcopy(self.data_list[idx])
        if self.pipeline is None:
            return data_info
        return self.pipeline(data_info)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a transformed data item, skipping invalid results.

        If the pipeline returns ``None`` (e.g. a tile with no
        annotations), the next item is tried instead.

        Args:
            idx: Item index.

        Returns:
            Transformed data dictionary.
        """
        data = self.prepare_data(idx)
        if data is None:
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)
        return data
