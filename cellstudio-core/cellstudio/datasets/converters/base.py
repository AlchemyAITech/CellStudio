from abc import ABC, abstractmethod
from typing import Any

from ...structures.csuos import CSUFeatureCollection

class BaseConverter(ABC):
    """Abstract base class for all Universal Data Foundation converters.
    
    Ensures that any external dataset format (COCO, YOLO, QuPath, etc.) 
    can be parsed into the UDF format and vice versa.
    """

    @abstractmethod
    def parse_to_udf(self, source_path: str, **kwargs) -> CSUFeatureCollection:
        """Parse external dataset format into UDF CSUFeatureCollection.
        
        Args:
            source_path: Path to the dataset root or annotation file.
            **kwargs: Extra format-specific arguments.
            
        Returns:
            A generic CSUFeatureCollection containing all image contexts and mapped features.
        """
        pass

    @abstractmethod
    def export_from_udf(self, udf_collection: CSUFeatureCollection, output_path: str, **kwargs) -> Any:
        """Export internal UDF format back to the external tool's format.
        
        Args:
            udf_collection: The internal representation.
            output_path: Path to dump the exported files.
            **kwargs: Extra format-specific arguments.
        """
        pass
