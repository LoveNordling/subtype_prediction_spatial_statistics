import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _configure_test_env(monkeypatch, tmp_path):
    # Matplotlib will otherwise try to use a non-writable config dir in this environment.
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))


@pytest.fixture
def stub_spatial_utils_module() -> types.ModuleType:
    module = types.ModuleType("spatial_utils")
    module.preprocess_external_cells = lambda df: df
    return module


@pytest.fixture
def stub_spatial_metrics_module() -> types.ModuleType:
    module = types.ModuleType("spatial_metrics")
    module.calculate_ripley_l = lambda *args, **kwargs: {}
    module.calculate_bidirectional_min_distance = lambda *args, **kwargs: {}
    module.calculate_newmans_assortativity = lambda *args, **kwargs: 0.0
    module.calculate_centrality_scores = lambda *args, **kwargs: {}
    module.calculate_cluster_cooccurrence_ratio = lambda *args, **kwargs: {}
    module.calculate_neighborhood_enrichment_test = lambda *args, **kwargs: {}
    module.calculate_objectobject_correlation = lambda *args, **kwargs: {}
    return module


@pytest.fixture
def load_script_module(repo_root, monkeypatch):
    def _load(script_name: str, module_name: str, stub_modules: dict[str, types.ModuleType] | None = None):
        if stub_modules:
            for name, module in stub_modules.items():
                monkeypatch.setitem(sys.modules, name, module)

        path = repo_root / script_name
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module spec for {path}")

        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
        return module

    return _load
