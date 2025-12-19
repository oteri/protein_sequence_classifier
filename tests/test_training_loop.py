import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import patch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from datasets import Dataset

# Add src to path to import esm2_classification
sys.path.append(str(Path(__file__).parents[1] / "src"))
from esm2_classification import train_model, FocalLoss, create_dataset

@pytest.fixture
def real_components(train_config):
    # Set seed for reproducibility to match experiment_results generation
    torch.manual_seed(train_config.get("seed", 42))
    
    model_name = train_config["model_name"]
    data_base_dir = Path(__file__).parent / "data"
    
    # 1. Prepare Data using create_dataset (like production)
    train_dataset, label_to_id = create_dataset(data_base_dir / "train")
    val_dataset, _ = create_dataset(data_base_dir / "validation", label_to_id)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_batch(batch):
        return tokenizer(batch["sequence"], padding="max_length", truncation=True, max_length=train_config["max_length"])
    
    train_dataset = train_dataset.map(tokenize_batch, batched=True, remove_columns=["sequence"])
    val_dataset = val_dataset.map(tokenize_batch, batched=True, remove_columns=["sequence"])
    
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    
    # Shuffle is True, so seed matters here
    train_dataloader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config["batch_size"])
    
    # 2. Prepare Model
    num_labels = len(label_to_id)
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=train_config["lora_r"],
        lora_alpha=train_config["lora_alpha"],
        lora_dropout=train_config["lora_dropout"],
        target_modules=["query", "key", "value", "dense"]
    )
    model = get_peft_model(model, peft_config)
    
    # 3. Optimizer & Accelerator
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"])
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * train_config["num_epochs"])
    
    accelerator = Accelerator()
    
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    return model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator

def test_training_loop_cross_entropy(real_components, experiment_results):
    model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator = real_components
    
    config = {
        "loss_type": "cross_entropy",
        "use_wandb": False,
        "num_epochs": 1
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
        "num_epochs": 1
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
        "num_epochs": 1
    }
    
    with patch("esm2_classification.logger") as mock_logger:
        history = train_model(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator, config=config)
        
        assert len(history) == 1
        # Should fallback to CE, so compare with CE results
        expected = experiment_results["cross_entropy"][0]
        
        assert history[0]["train_loss"] == pytest.approx(expected["train_loss"], rel=0.01)
        mock_logger.info.assert_any_call("Using default CrossEntropy Loss")

