#!/usr/bin/env python
import argparse
import os
import sys

# Ensure CellStudio package is resolvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cellstudio.datasets.converters.coco import COCOConverter
from cellstudio.datasets.converters.yolo import YOLOConverter
from cellstudio.datasets.converters.qupath import QuPathConverter
from cellstudio.datasets.converters.legacy import LegacyConverter

def parse_args():
    parser = argparse.ArgumentParser(description='Migrate external datasets into the CellStudio UDF format.')
    parser.add_argument('--format', type=str, required=True, choices=['coco', 'yolo', 'qupath', 'legacy'], 
                        help='The format of the source dataset.')
    parser.add_argument('--src', type=str, required=True, 
                        help='Source path to the annotation file (for COCO/QuPath) or image directory (for YOLO).')
    parser.add_argument('--out', type=str, required=True, 
                        help='Output path for the generated UDF JSON file.')
    parser.add_argument('--data-root', type=str, default="", 
                        help='Data root directory to resolving relative image paths.')
    
    # YOLO specific arguments
    parser.add_argument('--yolo-labels', type=str, 
                        help='Path to YOLO txt labels directory (required if format=yolo).')
    parser.add_argument('--yolo-classes', type=str, nargs='+', 
                        help='List of class names for YOLO in order of their class IDs.')
                        
    return parser.parse_args()

def main():
    args = parse_args()
    
    collection = None
    
    try:
        if args.format == 'coco':
            converter = COCOConverter()
            print(f"Migrating COCO dataset from {args.src} ...")
            collection = converter.parse_to_udf(args.src)
            
        elif args.format == 'yolo':
            if not args.yolo_labels or not args.yolo_classes:
                print("Error: --yolo-labels and --yolo-classes are required for YOLO format.")
                sys.exit(1)
            converter = YOLOConverter()
            print(f"Migrating YOLO dataset from {args.src} ...")
            collection = converter.parse_to_udf(args.src, labels_dir=args.yolo_labels, class_names=args.yolo_classes)
            
        elif args.format == 'qupath':
            converter = QuPathConverter()
            print(f"Migrating QuPath dataset from {args.src} ...")
            collection = converter.parse_to_udf(args.src)
            
        elif args.format == 'legacy':
            converter = LegacyConverter()
            print(f"Migrating Legacy CellStudio dataset from {args.src} ...")
            collection = converter.parse_to_udf(args.src, data_root=args.data_root)
        
        if collection:
            collection.save_json(args.out)
            print(f"Migration Success! Transformed {len(collection.features)} features across {len(collection.images)} images.")
            print(f"UDF Artifact saved to: {args.out}")
            
    except Exception as e:
        print(f"Migration Failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
