import pytest
from unittest.mock import patch
import sys
from pathlib import Path

# Add src to path to import esm2_classification
sys.path.append(str(Path(__file__).parents[1] / "src"))
from esm2_classification import train_model

def test_training_loop_cross_entropy(real_components, experiment_results):
    model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator = real_components
    
    config = {
        "loss_type": "cross_entropy",
        "use_wandb": False,
        "num_epochs": 1,
        "seed": 42,
        "deterministic": True
    }
    
    # Run for 1 epoch
    history = train_model(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator, config=config)
    
    assert len(history) == 1
    expected = experiment_results["cross_entropy"][0]
    
    assert history[0]["train_loss"] == pytest.approx(expected["train_loss"], rel=0.01)
    assert history[0]["val_loss"] == pytest.approx(expected["val_loss"], rel=0.01)

def test_training_loop_focal_loss(real_components, experiment_results):
    model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator = real_components
    
    config = {
        "loss_type": "focal",
        "focal_loss_gamma": 2.0,
        "use_wandb": False,
        "num_epochs": 1,
        "seed": 42,
        "deterministic": True
    }
    
    # Run for 1 epoch
    history = train_model(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator, config=config)
    
    assert len(history) == 1
    expected = experiment_results["focal"][0]
    
    assert history[0]["train_loss"] == pytest.approx(expected["train_loss"], rel=0.01)
    assert history[0]["val_loss"] == pytest.approx(expected["val_loss"], rel=0.01)

def test_training_loop_invalid_loss(real_components, experiment_results):
    model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator = real_components
    
    config = {
        "loss_type": "invalid_loss_name",
        "use_wandb": False,
        "num_epochs": 1,
        "seed": 42,
        "deterministic": True
    }
    
    with patch("esm2_classification.logger") as mock_logger:
        history = train_model(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator, config=config)
        
        assert len(history) == 1
        # Should fallback to CE, so compare with CE results
        expected = experiment_results["cross_entropy"][0]
        
        assert history[0]["train_loss"] == pytest.approx(expected["train_loss"], rel=0.01)
        mock_logger.info.assert_any_call("Using default CrossEntropy Loss")