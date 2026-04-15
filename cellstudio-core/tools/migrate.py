import json
import os
from pathlib import Path

def simplify_loop(loop, tolerance=1.0):
    try:
        from shapely.geometry import Polygon
        if len(loop) < 4:
            return [[round(x,1), round(y,1)] for x,y in loop]
        poly = Polygon(loop)
        simplified = poly.simplify(tolerance, preserve_topology=False)
        if simplified.geom_type == 'Polygon':
            return [[round(x,1), round(y,1)] for x, y in list(simplified.exterior.coords)]
        return [[round(x,1), round(y,1)] for x,y in loop]
    except ImportError:
        return [[round(x,1), round(y,1)] for x,y in loop]

def migrate_datasets(root_dir='e:/workspace/AlchemyTech/CellStudio/datasets'):
    for json_file in Path(root_dir).rglob('*.json'):
        if 'annotations' in json_file.parts:
            continue
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
            
        if not isinstance(data, dict): continue
        if 'feature_files' in data or 'features' not in data:
            continue # already migrated or not a monolithic dataset
            
        print(f"Migrating {json_file}...")
        
        features = data.pop('features')
        if not features:
            continue
            
        # Group and compress
        feat_map = {}
        for f in features:
            iid = f.get('image_id')
            if not iid: continue
            
            # Compress Polygon
            geom = f.get('geometry', {})
            if geom.get('type') == 'Polygon':
                coords = geom.get('coordinates', [])
                new_coords = []
                for loop in coords:
                    new_coords.append(simplify_loop(loop))
                geom['coordinates'] = new_coords
                
            if iid not in feat_map: feat_map[iid] = []
            feat_map[iid].append(f)
            
        base_dir = json_file.parent
        ann_dir = base_dir / 'annotations'
        ann_dir.mkdir(exist_ok=True)
        
        data['feature_files'] = {}
        for iid, feats in feat_map.items():
            rel_path = f"annotations/{iid}.json"
            out_path = base_dir / rel_path
            
            local = {
                "type": "FeatureCollection",
                "features": feats
            }
            with open(out_path, 'w', encoding='utf-8') as lf:
                json.dump(local, lf, indent=2, ensure_ascii=False)
                
            data['feature_files'][iid] = rel_path
            
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Finished {json_file} -> {len(feat_map)} split files.")

if __name__ == '__main__':
    migrate_datasets()
