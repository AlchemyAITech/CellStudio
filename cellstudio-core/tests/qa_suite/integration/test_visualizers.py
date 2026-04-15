import os
import pytest
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
qa_suite_dir = os.path.dirname(script_dir)
root_dir = os.path.dirname(os.path.dirname(qa_suite_dir))

# Wait for visualizers to stabilize in the future 
# Right now just load config and check if supports_vis flag behaves properly.

def test_visualizer_registry_integrity():
    from cellstudio.plotting.registry import PLOTTER_REGISTRY
    assert PLOTTER_REGISTRY is not None, "Plotter registry must be initialized."

def test_catalog_vis_flags():
    catalog_path = os.path.join(qa_suite_dir, "qa_catalog.yaml")
    with open(catalog_path, "r") as f:
        catalog = yaml.safe_load(f)["models"]
    
    for m in catalog:
        assert isinstance(m.get('supports_vis'), bool), f"{m['id']} missing supports_vis boolean."
