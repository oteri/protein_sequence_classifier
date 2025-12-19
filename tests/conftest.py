import pytest
import json
from pathlib import Path

@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def train_data_reference(fixtures_dir):
    with open(fixtures_dir / "train_data_reference.json") as f:
        return json.load(f)

@pytest.fixture
def experiment_results(fixtures_dir):
    with open(fixtures_dir / "experiment_results.json") as f:
        return json.load(f)

@pytest.fixture
def dummy_data_reference(fixtures_dir):
    path = fixtures_dir / "dummy_data_reference.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
