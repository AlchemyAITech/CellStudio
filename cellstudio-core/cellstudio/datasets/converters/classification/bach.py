import os
import glob
from pathstudio.datasets.converters.base import BaseDataConverter
from pathstudio.datasets.schema import CellDataItem, CellDatasetConfig

class BachConverter(BaseDataConverter):
    """
    Converter for the ICIAR2018 BACH Challenge dataset.
    Given its structure contains 'Photos' with subfolders as class names
    (e.g., Benign, InSitu, Invasive, Normal).
    """

    def convert(self) -> str:
        photos_dir = os.path.join(self.raw_data_dir, "Photos")
        if not os.path.exists(photos_dir):
            raise FileNotFoundError(f"Expected photos directory at {photos_dir}")

        items = []
        
        # In BACH, labels are typically folder names
        for label_dir in os.listdir(photos_dir):
            dir_path = os.path.join(photos_dir, label_dir)
            if not os.path.isdir(dir_path):
                continue
            
            # Sub-folders like 'Benign', 'Normal' are class labels
            class_name = label_dir.strip()
            
            img_paths = glob.glob(os.path.join(dir_path, "*.tif"))
            img_paths.extend(glob.glob(os.path.join(dir_path, "*.png")))
            img_paths.extend(glob.glob(os.path.join(dir_path, "*.jpg")))
            
            for img_path in img_paths:
                absolute_img_path = os.path.abspath(img_path)
                item = CellDataItem(
                    image_path=absolute_img_path,
                    cls_labels=[class_name],
                    metadata={"source": "BACH_Photos"}
                )
                items.append(item)
                
        dataset_config = CellDatasetConfig(items=items)
        # Note: WSI conversion is skipped for MVP logic. It requires patch extraction handling.
        
        return self.save_standardized_json(dataset_config, "bach_standard.json")
