import os
import json

datasets_dir = r"e:\workspace\AlchemyTech\CellStudio\datasets"
tree = {}

for root, dirs, files in os.walk(datasets_dir):
    rel_path = os.path.relpath(root, datasets_dir)
    if rel_path == ".":
        continue
    
    parts = rel_path.split(os.sep)
    current = tree
    for part in parts:
        if part not in current:
            current[part] = {"_files": 0, "_subdirs": {}}
        current = current[part]["_subdirs"]
    
    # Re-traverse to add file count to current dir
    current_node = tree
    for part in parts[:-1]:
         current_node = current_node[part]["_subdirs"]
    current_node[parts[-1]]["_files"] = len(files)

# summarize top level
summary = {}
for task_type in tree:
    summary[task_type] = {}
    for dataset_name in tree[task_type]["_subdirs"]:
        summary[task_type][dataset_name] = {}
        for split in tree[task_type]["_subdirs"][dataset_name]["_subdirs"]:
            files_count = tree[task_type]["_subdirs"][dataset_name]["_subdirs"][split]["_files"]
            summary[task_type][dataset_name][split] = files_count

out_path = os.path.join(datasets_dir, "dataset_summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=4)
print(f"Summary saved to {out_path}")
