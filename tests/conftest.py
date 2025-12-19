import pytest
import json
import torch
import sys
from pathlib import Path
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader

# Add src to path to import esm2_classification
sys.path.append(str(Path(__file__).parents[1] / "src"))
from esm2_classification import create_dataset

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
def esm2_model(train_config, train_data_reference):
    model_name = train_config["model_name"]
    label_to_id = {'Cas1': 0, 'Cas12': 1, 'Cas2': 2}
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
    return model


@pytest.fixture
def real_components(train_config, esm2_model):
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
    # 2. Model is provided by fixture
    
    # 3. Optimizer & Accelerator
    optimizer = torch.optim.AdamW(esm2_model.parameters(), lr=train_config["learning_rate"])
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * train_config["num_epochs"])
    
    accelerator = Accelerator()
    
    esm2_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        esm2_model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    return esm2_model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator

@pytest.fixture
def dummy_data_reference(fixtures_dir):
    path = fixtures_dir / "dummy_data_reference.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
