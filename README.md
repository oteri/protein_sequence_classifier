# Protein Sequence Classifier

## Overview

This repository provides a **modular and extensible framework** for protein **sequence classification** using **ESM2** models fine-tuned with **LoRA** adapters.
The framework is built for both **human developers** and **coding agents**, with clear boundaries between configuration, data handling, model definition, and training execution.

All **ESM2 variants available on Hugging Face** are supported through configuration:

* `esm2_t6_8M`
* `esm2_t12_35M`
* `esm2_t30_150M`
* `esm2_t33_650M`
* (or any other HF-hosted ESM2 variant)

Label assignment is automatic and derived from filenames.

---

## Dataset Structure

The project expects the following directory layout:

```
data/
    train/
        <LabelA>.fasta
        <LabelB>.fasta
        ...
    validation/
        <LabelA>.fasta
        <LabelB>.fasta
        ...
    test/
        <LabelA>.fasta
        <LabelB>.fasta
        ...
```

### Label extraction rule

The **label** for all sequences inside a FASTA file corresponds to the **filename without extension**.

Example:

```
data/train/Cas12.fasta  →  label = "Cas12"
```

Each FASTA file must contain sequences belonging to a **single label**.

This behavior is implemented in `esm2_classification.py`.

---

## Project Structure

```
configs/
    train.yaml          # model & training hyperparameters
    accelerate.yaml     # hardware & distribution config
data/
src/
    esm2_classification.py   # main training and classification logic
Makefile
```

Key file:

* **`esm2_classification.py`**
  Contains dataset parsing, label assignment, ESM2 loading from Hugging Face, LoRA injection, tokenization, training loop, and evaluation pipeline.

---

## Requirements

Python version target: **3.12**
Exact dependencies depend on the training setup and will be tested progressively.

A minimal environment includes:

* `torch`
* `transformers`
* `accelerate`
* `peft` (for LoRA)
* `datasets` (if used by your pipeline)
* `biopython` (FASTA parsing)
* `pyyaml`

Dependencies are managed via `pyproject.toml` using `uv`.

---

## Training Workflow

### 1. Prepare Dataset

Fill the folders:

```
data/train/
data/validation/
data/test/
```

Each `.fasta` file defines one class.

### 2. Configure Accelerate

Edit:

```
configs/accelerate.yaml
```

This file defines:

* GPU / multi-GPU setup
* mixed precision (fp16/bf16)
* distributed strategy

### 3. Configure Training

Edit:

```
configs/train.yaml
```

Configurable elements include:

* which **ESM2** model to use (Hugging Face ID)
* LoRA hyperparameters (`r`, `alpha`, `dropout`)
* training batch size
* maximum sequence length
* learning rate, optimizer
* number of epochs
* output directories

### 4. Run Training

Use:

```bash
make train
```

This executes:

```
accelerate launch src/esm2_classification.py
```

with configs automatically loaded.

---

## Inference (COMING SOON)

Inference support will include:

* loading the fine-tuned checkpoint
* running prediction on a FASTA file or raw sequence
* producing label probabilities

The README will be updated once the inference API is implemented.

---

## Extensibility

This framework is intentionally designed to be adaptable:

You can replace or extend:

* tokenization logic
* LoRA configuration
* dataset structure
* training loop (e.g., curriculum, early stopping)
* evaluation metrics
* model selection
* logging and tracking (e.g., WandB, TensorBoard)

