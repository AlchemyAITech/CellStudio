import os
import pandas as pd
from pathstudio.datasets.converters.base import BaseDataConverter
from pathstudio.datasets.schema import CellDataItem, CellDatasetConfig

class Midog25Converter(BaseDataConverter):
    """
    Converter for the MIDOG 25 classification dataset.
    Based on 'MIDOG25_Atypical_Classification_Train_Set.csv'
    where each row has 'image_id' (patch filename), 'filename' (WSI name),
    and 'majority' (e.g., NMF, AMF) as the ground truth class.
    """
    
    def convert(self) -> str:
        csv_path = os.path.join(self.raw_data_dir, "MIDOG25_Atypical_Classification_Train_Set.csv")
        images_dir = os.path.join(self.raw_data_dir, "MIDOG25_Binary_Classification_Train_Set") # Assuming patch folder name
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Annotation CSV not found at {csv_path}")
            
        df = pd.read_csv(csv_path)
        items = []
        
        for _, row in df.iterrows():
            img_filename = str(row['image_id'])
            
            # The exact folder might differ, but this provides the structure
            img_path = os.path.abspath(os.path.join(images_dir, img_filename))
            
            # If the patch doesn't exist yet, we still record the schema, 
            # Or we can skip. Better to record since it's an abstract index.
            label = str(row['majority']).strip()
            if pd.isna(row['majority']) or label == "nan":
                continue # Skip unlabeled
                
            metadata = {
                "source": "MIDOG25",
                "wsi_filename": str(row.get('filename', '')),
                "tumor_type": str(row.get('Tumor', '')),
                "expert_votes": {
                    "e1": str(row.get('expert1', '')),
                    "e2": str(row.get('expert2', '')),
                    "e3": str(row.get('expert3', ''))
                }
            }
            
            # Optionally add coordinate as metadata or as a bbox if we want detection 
            # But the task says Classification.
            if 'coordinateX' in row and 'coordinateY' in row:
                metadata['center_x'] = row['coordinateX']
                metadata['center_y'] = row['coordinateY']
                
            items.append(CellDataItem(
                image_path=img_path,
                cls_labels=[label],
                metadata=metadata
            ))

        dataset_config = CellDatasetConfig(items=items)
        return self.save_standardized_json(dataset_config, "midog25_standard.json")
