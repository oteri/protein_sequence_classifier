import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import yaml
    import torch
    import logging
    import argparse
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
    from peft import get_peft_model, LoraConfig, TaskType
    from accelerate import Accelerator
    from datasets import Dataset
    from Bio import SeqIO
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import numpy as np

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    return (
        Accelerator,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataLoader,
        LoraConfig,
        Path,
        SeqIO,
        TaskType,
        accuracy_score,
        get_peft_model,
        get_scheduler,
        logger,
        precision_recall_fscore_support,
        torch,
        yaml,
    )


@app.cell
def _(Path, SeqIO, logger, yaml):
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
            logger.warning(f"No fasta/faa files found in {folder_path}")

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
            logger.warning(f"Directory {folder_path} does not exist.")
            return None, None

        sequences, labels_text = load_data_from_folder(folder_path)

        if not sequences:
            logger.warning(f"No sequences collected from {folder_path}")
            return None, None

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

        from datasets import Dataset # Import here to ensure visibility if needed or rely on outer scope
        dataset = Dataset.from_dict({"sequence": filtered_sequences, "label": labels_id})
        return dataset, label_to_id
    return create_dataset, load_config


@app.cell
def _(Accelerator, load_config):
    # Simulate args
    config_file = "configs/train.yaml"

    # 1. Initialize Accelerator
    accelerator = Accelerator()
    print("Accelerator initialized")

    # 2. Load Config
    config = load_config(config_file)
    print(f"Loaded config: {config}")
    return accelerator, config


@app.cell
def _(Path, create_dataset):
    # 3. Load Data & Create Label Map
    print("Loading training data...")
    train_dataset, label_to_id = create_dataset("data/train")
    if train_dataset is None:
        raise ValueError("Training data is empty! Please populate data/train.")

    id_to_label = {v: k for k, v in label_to_id.items()}
    num_labels = len(label_to_id)
    print(f"Found {num_labels} classes: {label_to_id}")

    print("Loading validation and test data...")
    val_path = "data/validation" if Path("data/validation").exists() else "data/validate"
    val_dataset, _ = create_dataset(val_path, label_to_id)
    test_dataset, _ = create_dataset("data/test", label_to_id)
    return (
        id_to_label,
        label_to_id,
        num_labels,
        test_dataset,
        train_dataset,
        val_dataset,
    )


@app.cell
def _(
    AutoTokenizer,
    accelerator,
    config,
    test_dataset,
    train_dataset,
    val_dataset,
):
    # 4. Tokenization
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_batch(batch):
        return tokenizer(batch["sequence"], padding="max_length", truncation=True, max_length=config["max_length"])

    with accelerator.main_process_first():
        train_dataset_tokenized = train_dataset.map(tokenize_batch, batched=True, remove_columns=["sequence"])
        val_dataset_tokenized = val_dataset.map(tokenize_batch, batched=True, remove_columns=["sequence"]) if val_dataset else None
        test_dataset_tokenized = test_dataset.map(tokenize_batch, batched=True, remove_columns=["sequence"]) if test_dataset else None

    train_dataset_tokenized.set_format("torch")
    if val_dataset_tokenized: val_dataset_tokenized.set_format("torch")
    if test_dataset_tokenized: test_dataset_tokenized.set_format("torch")
    return (
        model_name,
        test_dataset_tokenized,
        tokenizer,
        train_dataset_tokenized,
        val_dataset_tokenized,
    )


@app.cell
def _(
    DataLoader,
    config,
    test_dataset_tokenized,
    train_dataset_tokenized,
    val_dataset_tokenized,
):
    # 5. DataLoaders
    batch_size = config["batch_size"]
    train_dataloader = DataLoader(train_dataset_tokenized, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset_tokenized, batch_size=batch_size) if val_dataset_tokenized else None
    test_dataloader = DataLoader(test_dataset_tokenized, batch_size=batch_size) if test_dataset_tokenized else None
    return test_dataloader, train_dataloader, val_dataloader


@app.cell
def _(
    AutoModelForSequenceClassification,
    LoraConfig,
    TaskType,
    accelerator,
    config,
    get_peft_model,
    id_to_label,
    label_to_id,
    model_name,
    num_labels,
):
    # 6. Model
    print(f"Loading model {model_name}...")
    _model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id
    )

    # 7. LoRA
    print("Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=config["lora_r"], 
        lora_alpha=config["lora_alpha"], 
        lora_dropout=config["lora_dropout"],
        target_modules=["query", "key", "value", "dense"] 
    )
    model_lora = get_peft_model(_model, peft_config)
    if accelerator.is_local_main_process:
        model_lora.print_trainable_parameters()
    return (model_lora,)


@app.cell
def _(config, get_scheduler, model_lora, torch, train_dataloader):
    # 8. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model_lora.parameters(), lr=config["learning_rate"])

    num_epochs = config["num_epochs"]
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler, num_epochs, optimizer


@app.cell
def _(
    accelerator,
    lr_scheduler,
    model_lora,
    optimizer,
    test_dataloader,
    train_dataloader,
    val_dataloader,
):
    # 9. Prepare
    # Use new variable names to avoid marimo global variable redefinition conflict
    p_model, p_optimizer, p_train_dataloader, p_val_dataloader, p_lr_scheduler = accelerator.prepare(
        model_lora, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    if test_dataloader:
        p_test_dataloader = accelerator.prepare(test_dataloader)
    else:
        p_test_dataloader = None
    return (
        p_lr_scheduler,
        p_model,
        p_optimizer,
        p_test_dataloader,
        p_train_dataloader,
        p_val_dataloader,
    )


@app.cell
def _(
    accelerator,
    accuracy_score,
    num_epochs,
    p_lr_scheduler,
    p_model,
    p_optimizer,
    p_train_dataloader,
    p_val_dataloader,
    torch,
):
    # 10. Training Loop
    def run_training_loop():
        print("Starting training...")
        for epoch in range(num_epochs):
            p_model.train()
            total_loss = 0
            for batch in p_train_dataloader:
                outputs = p_model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                p_optimizer.step()
                p_lr_scheduler.step()
                p_optimizer.zero_grad()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(p_train_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

            # Validation
            if p_val_dataloader:
                p_model.eval()
                val_loss = 0
                all_preds = []
                all_labels = []
                for batch in p_val_dataloader:
                    with torch.no_grad():
                        outputs = p_model(**batch)
                        val_loss += outputs.loss.item()
                        predictions = outputs.logits.argmax(dim=-1)
                        preds, labels = accelerator.gather_for_metrics((predictions, batch["label"]))
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                avg_val_loss = val_loss / len(p_val_dataloader)
                accuracy = accuracy_score(all_labels, all_preds)
                print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {avg_val_loss:.4f} - Val Accuracy: {accuracy:.4f}")

    run_training_loop()
    return


@app.cell
def _(
    accelerator,
    accuracy_score,
    p_model,
    p_test_dataloader,
    precision_recall_fscore_support,
    torch,
):
    # 11. Final Evaluation on Test Set
    def run_test_loop():
        if p_test_dataloader:
            print("Evaluating on test set...")
            p_model.eval()
            all_preds = []
            all_labels = []
            for batch in p_test_dataloader:
                with torch.no_grad():
                    outputs = p_model(**batch)
                    predictions = outputs.logits.argmax(dim=-1)
                    preds, labels = accelerator.gather_for_metrics((predictions, batch["label"]))
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

            print(f"Test Results: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    run_test_loop()
    return


@app.cell
def _(Path, accelerator, config, label_to_id, p_model, tokenizer, yaml):
    # 12. Save Model
    output_dir = config["output_dir"]
    if output_dir:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(p_model)
        unwrapped_model.save_pretrained(output_dir, is_main_process=accelerator.is_main_process)
        tokenizer.save_pretrained(output_dir)
        if accelerator.is_main_process:
            # Save label map
            with open(Path(output_dir) / "label_map.yaml", "w") as f:
                yaml.dump(label_to_id, f)
            print(f"Model saved to {output_dir}")
    return


if __name__ == "__main__":
    app.run()
