import os
import csv
import sys

# Ensure CellStudio is in path if run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from cellstudio.datasets.converters.base import BaseDataConverter
from cellstudio.datasets.schema import CellDataItem, CellDatasetConfig

class MidoClassificationConverter(BaseDataConverter):
    """
    Converts MIDOG binary classification CSVs into Standardized CellDatasetConfig JSONs.
    """
    
    def convert(self) -> str:
        splits_dir = os.path.join(self.raw_data_dir, "splits")
        img_dir = os.path.join(self.raw_data_dir, "MIDOG25_Binary_Classification_Train_Set")
        
        for split in ['train', 'val', 'test']:
            csv_path = os.path.join(splits_dir, f"{split}.csv")
            if not os.path.exists(csv_path):
                continue
                
            items = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_id = row['image_id']
                    label = row.get('majority')
                    if not label:
                        continue
                    
                    img_path = os.path.abspath(os.path.join(img_dir, image_id))
                    
                    item = CellDataItem(
                        image_path=img_path,
                        cls_labels=[label],
                        metadata={"source": "MIDOG", "expert1": row.get('expert1', '')}
                    )
                    items.append(item)
                    
            config = CellDatasetConfig(items=items)
            out_filename = f"mido_cls_{split}.json"
            self.save_standardized_json(config, out_filename)
            
        return self.output_dir

if __name__ == '__main__':
    converter = MidoClassificationConverter(
        raw_data_dir='e:/workspace/AlchemyTech/CellStudio/datasets/classfication/MIDOG',
        output_dir='e:/workspace/AlchemyTech/CellStudio/datasets/classfication/MIDOG/standardized'
    )
    converter.convert()
