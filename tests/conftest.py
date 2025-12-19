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
def train_config():
    return {
        "model_name": "facebook/esm2_t6_8M_UR50D",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0, # Disable dropout for reproducibility
        "batch_size": 4, 
        "max_length": 128,
        "learning_rate": 0.0005,
        "num_epochs": 1,
        "output_dir": "./output",
        "seed": 42,
        "deterministic": True, # Enable deterministic mode
        "label_smoothing": 0.1,
        "loss_type": "focal",
        "focal_loss_gamma": 2.0,
        "use_wandb": False,
        "wandb_entity": "francesco-oteri-perso-sorbonne-universit-",
        "wandb_project": "protein-sequence-classifier",
        "wandb_run_name": "esm2-lora-run"
    }

@pytest.fixture
def dummy_data_reference(fixtures_dir):
    path = fixtures_dir / "dummy_data_reference.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
