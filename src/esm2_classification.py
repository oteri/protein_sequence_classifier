from pathlib import Path
import yaml
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

def load_config(config_path="configs/train.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data_from_folder(folder_path):
    sequences = []
    labels = []
    # Get all fasta files (support .fasta and .faa)
    folder = Path(folder_path)
    fasta_files = list(folder.glob("*.fasta")) + list(folder.glob("*.faa"))

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

def train_model(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator, num_epochs, config=None):
    logger.info("Starting training...")
    use_wandb = config.get("use_wandb", False) if config else False

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_dataloader):
            if i == 0:
                 # Debug print
                if 'labels' not in batch:
                     logger.warning(f"WARNING: 'labels' key missing in batch! Keys: {list(batch.keys())}")

            outputs = model(**batch)

            loss = outputs.loss
            if loss is None:
                 raise ValueError(f"Model return None loss. Batch keys: {list(batch.keys())}")

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if use_wandb and accelerator.is_main_process:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/step": epoch * len(train_dataloader) + i
                })

        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

        if use_wandb and accelerator.is_main_process:
            wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch + 1})

        # Validation
        if val_dataloader:
            val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_dataloader, accelerator)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f}")
            
            if use_wandb and accelerator.is_main_process:
                wandb.log({
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/precision": val_precision,
                    "val/recall": val_recall,
                    "val/f1": val_f1,
                    "epoch": epoch + 1
                })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    # 1. Initialize Accelerator
    accelerator = Accelerator()
    logger.info("Accelerator initialized")

    # 2. Load Config
    config = load_config(args.config_file)
    logger.info(f"Loaded config: {config}")

    # 3. Load Data & Create Label Map
    logger.info("Loading training data...")
    train_path = "data/train"
    val_path = "data/validation" if Path("data/validation").exists() else "data/validate"
    test_path = "data/test"

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
            project=config.get("wandb_project", "protein-sequence-classifier"),
            name=config.get("wandb_run_name", None),
            config=config
        )

    # 11. Training Loop
    train_model(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, accelerator, num_epochs, config=config)

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
