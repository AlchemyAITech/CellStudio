import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

@dataclass
class BBox:
    """Bounding box format for objects."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str
    confidence: Optional[float] = None

@dataclass
class Polygon:
    """Segmentation coordinates format."""
    label: str
    points: List[float] = field(default_factory=list)

@dataclass
class CellDataItem:
    """Standardized wrapper representing a single image data frame."""
    image_path: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    
    cls_labels: List[str] = field(default_factory=list)
    bboxes: List[BBox] = field(default_factory=list)
    polygons: List[Polygon] = field(default_factory=list)
    mask_path: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)

class CellDatasetConfig:
    """Standard wrapper to serialize list of items into JSON."""
    def __init__(self, items: List[CellDataItem] = None, classes: List[str] = None):
        self.items = items or []
        self._classes = classes or []
        
    @property
    def classes(self) -> List[str]:
        if not self._classes:
            cls_set = set()
            for item in self.items:
                cls_set.update(item.cls_labels)
                cls_set.update([b.label for b in item.bboxes])
                cls_set.update([p.label for p in item.polygons])
            self._classes = sorted(list(cls_set))
        return self._classes

    def save(self, filepath: str):
        data = {
            "classes": self.classes,
            "items": [asdict(item) for item in self.items]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "CellDatasetConfig":
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        classes = data.get("classes", [])
        items_data = data.get("items", [])
        
        items = []
        for d in items_data:
            bboxes = [BBox(**b) for b in d.get("bboxes", [])]
            polygons = [Polygon(**p) for p in d.get("polygons", [])]
            
            item = CellDataItem(
                image_path=d["image_path"],
                image_width=d.get("image_width"),
                image_height=d.get("image_height"),
                cls_labels=d.get("cls_labels", []),
                bboxes=bboxes,
                polygons=polygons,
                mask_path=d.get("mask_path"),
                metadata=d.get("metadata", {})
            )
            items.append(item)
            
        return cls(items=items, classes=classes)
