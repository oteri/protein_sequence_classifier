from pathlib import Path
import yaml
import time
from tqdm.auto import tqdm
import torch
import logging
import argparse
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from datasets import Dataset
from Bio import SeqIO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os
import random

import torch.nn as nn
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

def seed_everything(seed: int, deterministic: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None:
             self.alpha = self.alpha.to(inputs.device)
             
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def load_config(config_path="configs/train.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data_from_folder(folder_path):
    sequences = []
    labels = []
    # Get all fasta files (support .fasta and .faa)
    folder = Path(folder_path)
    fasta_files = sorted(list(folder.glob("*.fasta"))) + sorted(list(folder.glob("*.faa")))

    if not fasta_files:
        raise ValueError(f"No fasta/faa files found in {folder_path}")

    for file_path in fasta_files:
        # Extract label from filename (e.g., "Cas12.fasta" -> "Cas12")
        label = file_path.stem

        # Read sequences
        try:
            records = list(SeqIO.parse(file_path, "fasta"))
            if not records:
                logger.warning(f"No sequences found in {file_path}")
                continue
            for record in records:
                sequences.append(str(record.seq))
                labels.append(label)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    return sequences, labels

def create_dataset(folder_path, label_to_id=None):
    if not Path(folder_path).exists():
        raise ValueError(f"Directory {folder_path} does not exist.")

    sequences, labels_text = load_data_from_folder(folder_path)

    if not sequences:
        raise ValueError(f"No sequences collected from {folder_path}")

    if label_to_id is None:
        unique_labels = sorted(list(set(labels_text)))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}

    # Filter labels
    labels_id = []
    filtered_sequences = []
    for seq, label in zip(sequences, labels_text):
        if label in label_to_id:
            labels_id.append(label_to_id[label])
            filtered_sequences.append(seq)
        else:
            logger.warning(f"Label {label} not in label map. Skipping sequence.")

    dataset = Dataset.from_dict({"sequence": filtered_sequences, "labels": labels_id})
    return dataset, label_to_id

def print_dataset_statistics(train_ds, val_ds, test_ds, id_to_label):
    from collections import Counter
    
    splits = ["Train", "Validation", "Test"]
    datasets = [train_ds, val_ds, test_ds]
    
    stats = {split: Counter(ds["labels"]) for split, ds in zip(splits, datasets)}
    label_ids = sorted(id_to_label.keys())
    
    headers = ["Label", "Train", "Validation", "Test", "Total"]
    table_rows = []
    
    col_totals = {split: 0 for split in splits}
    grand_total = 0
    
    for label_id in label_ids:
        label_name = id_to_label[label_id]
        row = [label_name]
        row_total = 0
        for split in splits:
            count = stats[split].get(label_id, 0)
            row.append(count)
            row_total += count
            col_totals[split] += count
        row.append(row_total)
        grand_total += row_total
        table_rows.append(row)
        
    totals_row = ["TOTAL"]
    for split in splits:
        totals_row.append(col_totals[split])
    totals_row.append(grand_total)
    table_rows.append(totals_row)
    
    # Calculate column widths
    all_data = [headers] + table_rows
    col_widths = [max(len(str(item)) for item in col) for col in zip(*all_data)]
    col_widths = [w + 2 for w in col_widths]

    # Build table string
    lines = ["\nDataset Statistics:"]
    header_str = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_str)
    lines.append("-" * len(header_str))
    
    for row in table_rows:
        row_str = "".join(str(val).ljust(w) for val, w in zip(row, col_widths))
        lines.append(row_str)
    
    logger.info("\n".join(lines))

def evaluate(model, dataloader, accelerator):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            if outputs.loss is not None:
                total_loss += outputs.loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            preds, labels = accelerator.gather_for_metrics((predictions, batch["labels"]))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, accuracy, precision, recall, f1

def train_model(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator, config):

    seed_everything(config.get("seed", 42), config.get("deterministic", False))
        
    logger.info("Starting training...")
    use_wandb = config.get("use_wandb", False) if config else False
    
    loss_type = config.get("loss_type", "cross_entropy")
    focal_gamma = config.get("focal_loss_gamma", 2.0)
    
    loss_fct = None
    if loss_type == "focal":
        logger.info(f"Using Focal Loss with gamma={focal_gamma}")
        loss_fct = FocalLoss(gamma=focal_gamma)
    else:
        logger.info("Using default CrossEntropy Loss")

    history = []
    num_epochs = config["num_epochs"]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_losses = []
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        for i, batch in progress_bar:
            batch_start_time = time.time()
            outputs = model(**batch)

            if loss_type == "focal":
                loss = loss_fct(outputs.logits, batch["labels"])
            else:
                loss = outputs.loss

            if loss is None:
                 raise ValueError(f"Model return None loss. Batch keys: {list(batch.keys())}")

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            loss_val = loss.item()
            total_loss += loss_val
            batch_losses.append(loss_val)

            batch_duration = time.time() - batch_start_time
            progress_bar.set_postfix(loss=loss_val)

            if use_wandb and accelerator.is_main_process:
                wandb.log({
                    "train/batch_loss": loss_val,
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/step": epoch * len(train_dataloader) + i,
                    "train/batch_time": batch_duration
                })

        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

        if use_wandb and accelerator.is_main_process:
            wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch + 1})

        # Validation
        val_metrics = {}
        if val_dataloader:
            val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_dataloader, accelerator)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f}")
            
            val_metrics = {
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1
            }

            if use_wandb and accelerator.is_main_process:
                wandb_log_data = {f"val/{k}": v for k, v in val_metrics.items()} # map keys
                wandb_log_data["epoch"] = epoch + 1
                wandb.log(wandb_log_data)
        
        # Record history
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "batch_losses": batch_losses,
            **val_metrics
        }
        history.append(epoch_record)

    return history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to the config file.")
    parser.add_argument("--dataset_root", type=str, help="Root folder for the datasets.")
    args = parser.parse_args()

    # 1. Initialize Accelerator
    accelerator = Accelerator()
    logger.info("Accelerator initialized")

    # 2. Load Config
    config = load_config(args.config_file)
    
    # Override config with command line argument if provided
    if args.dataset_root:
        config["dataset_root"] = args.dataset_root
    elif "dataset_root" not in config:
        config["dataset_root"] = "data"
        
    logger.info(f"Loaded config: {config}")

    # 3. Load Data & Create Label Map
    logger.info(f"Loading data from {config['dataset_root']}...")
    dataset_root = Path(config["dataset_root"])
    train_path = dataset_root / "train"
    val_path = dataset_root / "validation" if (dataset_root / "validation").exists() else dataset_root / "validate"
    test_path = dataset_root / "test"

    try:
        train_dataset, label_to_id = create_dataset(train_path)
    except ValueError:
        raise ValueError(f"Training data is empty! Please populate {train_path}.")

    id_to_label = {v: k for k, v in label_to_id.items()}
    num_labels = len(label_to_id)
    logger.info(f"Found {num_labels} classes: {label_to_id}")

    logger.info(f"Loading validation and test data from {val_path} and {test_path}")

    try:
        val_dataset, _ = create_dataset(val_path, label_to_id)
    except ValueError:
        raise ValueError(f"Validation data is empty! Please populate {val_path}.")

    try:
        test_dataset, _ = create_dataset(test_path, label_to_id)
    except ValueError:
         raise ValueError(f"Test data is empty! Please populate {test_path}.")

    if accelerator.is_main_process:
        print_dataset_statistics(train_dataset, val_dataset, test_dataset, id_to_label)

    # 4. Tokenization
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_batch(batch):
        return tokenizer(batch["sequence"], padding="max_length", truncation=True, max_length=config["max_length"])

    with accelerator.main_process_first():
        train_dataset_tokenized = train_dataset.map(tokenize_batch, batched=True, remove_columns=["sequence"])
        val_dataset_tokenized = val_dataset.map(tokenize_batch, batched=True, remove_columns=["sequence"])
        test_dataset_tokenized = test_dataset.map(tokenize_batch, batched=True, remove_columns=["sequence"])

    train_dataset_tokenized.set_format("torch")
    val_dataset_tokenized.set_format("torch")
    test_dataset_tokenized.set_format("torch")

    # 5. DataLoaders
    batch_size = config["batch_size"]
    train_dataloader = DataLoader(train_dataset_tokenized, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset_tokenized, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset_tokenized, batch_size=batch_size)

    # 6. Model
    logger.info(f"Loading model {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id
    )

    # 7. LoRA
    logger.info("Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=["query", "key", "value", "dense"]
    )
    model = get_peft_model(model, peft_config)
    if accelerator.is_local_main_process:
        model.print_trainable_parameters()

    # 8. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    num_epochs = config["num_epochs"]
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # 9. Prepare
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    if test_dataloader:
        test_dataloader = accelerator.prepare(test_dataloader)

    # 10. Initialize wandb
    if config.get("use_wandb", False) and accelerator.is_main_process:
        wandb.init(
            entity=config.get("wandb_entity",  None),
            project=config.get("wandb_project",  None),
            name=config.get("wandb_run_name", None),
            config=config
        )

    # 11. Training Loop
    train_model(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator, config=config)

    # 12. Final Evaluation on Test Set
    if test_dataloader:
        logger.info("Evaluating on test set...")
        _, accuracy, precision, recall, f1 = evaluate(model, test_dataloader, accelerator)
        logger.info(f"Test Results: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if config.get("use_wandb", False) and accelerator.is_main_process:
            wandb.log({
                "test/accuracy": accuracy,
                "test/precision": precision,
                "test/recall": recall,
                "test/f1": f1
            })

    # 13. Save Model
    output_dir = config["output_dir"]
    if output_dir:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, is_main_process=accelerator.is_main_process)
        tokenizer.save_pretrained(output_dir)
        if accelerator.is_main_process:
            # Save label map
            with open(Path(output_dir) / "label_map.yaml", "w") as f:
                yaml.dump(label_to_id, f)
            logger.info(f"Model saved to {output_dir}")
            
            if config.get("use_wandb", False):
                wandb.finish()

if __name__ == "__main__":
    main()
