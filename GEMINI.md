# Gemini Agent Project Guide: Protein Sequence Classifier

This document provides a concise guide for the Gemini coding agent to effectively understand and work on this project.

## 1. Project Overview

This project is a modular framework for protein sequence classification. It uses ESM2 models with LoRA adapters for fine-tuning. The goal is to classify protein sequences from FASTA files, where the filename determines the label.

## 2. Key Files & Directories

-   `src/esm2_classification.py`: This is the main script that will contain the core logic for data loading, model training, and evaluation.
-   `configs/train.yaml`: This file will contain all hyperparameters for training, such as the ESM2 model to use, LoRA parameters, batch size, learning rate, etc.
-   `configs/accelerate.yaml`: This file is for configuring the hardware and distributed training setup (e.g., GPUs, mixed precision).
-   `data/`: This directory contains the training, validation, and test datasets, organized in subdirectories.
-   `Makefile`: This file will contain helper commands for common tasks like training.

## 3. Core Logic (`src/esm2_classification.py`)

The `esm2_classification.py` script should implement the following logic:

1.  **Load Configuration:** Read the `train.yaml` and `accelerate.yaml` files.
2.  **Dataset Loading:**
    *   Scan the `data/train`, `data/validation`, and `data/test` directories.
    *   For each `.fasta` file, extract the label from the filename (e.g., `Cas12.fasta` -> `Cas12`).
    *   Load the protein sequences from the FASTA files.
3.  **Model Loading:**
    *   Load the specified ESM2 model from Hugging Face.
    *   Inject LoRA adapters into the model using the `peft` library.
4.  **Tokenization:** Tokenize the protein sequences for the ESM2 model.
5.  **Training:**
    *   Use the `accelerate` library to handle the training loop.
    *   Train the model on the training dataset.
    *   Evaluate the model on the validation dataset.
6.  **Evaluation:**
    *   After training, evaluate the final model on the test dataset.
    *   Save the trained model and evaluation results.

## 4. Configuration

### `configs/train.yaml`

This file should define the following parameters:

-   `model_name`: The Hugging Face model ID of the ESM2 model (e.g., `facebook/esm2_t6_8M_UR50D`).
-   `lora_r`: The `r` parameter for LoRA.
-   `lora_alpha`: The `alpha` parameter for LoRA.
-   `lora_dropout`: The dropout rate for LoRA.
-   `batch_size`: The training batch size.
-   `max_length`: The maximum sequence length.
-   `learning_rate`: The learning rate for the optimizer.
-   `num_epochs`: The number of training epochs.
-   `output_dir`: The directory to save the trained model and results.

### `configs/accelerate.yaml`

This file is configured by the user to match their hardware setup. It controls distributed training and mixed precision.

## 5. Commands

The primary command to run training is:

```bash
make train
```

This command should execute:

```bash
accelerate launch src/esm2_classification.py
```

## 6. Dataset Structure

The data should be organized as follows:

```
data/
├── train/
│   ├── <LabelA>.fasta
│   └── <LabelB>.fasta
├── validation/
│   ├── <LabelA>.fasta
│   └── <LabelB>.fasta
└── test/
    ├── <LabelA>.fasta
    └── <LabelB>.fasta
```

Each `.fasta` file contains sequences for a single class, and the filename (without extension) is used as the class label.

## 7. Coding Best Practices for Gemini Agent:

**General Principles:**
-   **Clarification:** Always ask clarifying questions to the user if the task is ambiguous or requires more detail before proceeding.
-   **Technology Preference:** When not explicitly specified, prioritize the following:
    -   **Machine Learning:** `huggingface` and `pytorch`.
    -   **Data Analysis:** `polars` over `pandas`.
    -   **Path manipulation:** `pathlib.Path`.
    -   **Testing:** `pytest`.
    -   **Package Management:** Use `uv`  to add/delete/update packages .

**Implementation Workflow:**
1.  **Plan:** Before writing any code, formulate a clear, step-by-step plan.
2.  **Information Gathering:** Utilize available tools (`read_file`, `list_directory`, `search_file_content`, `codebase_investigator`, `google_web_search`) to gather necessary documentation and understand the existing codebase.
3.  **Code Reusability:** Prioritize reusing existing code. Avoid rewriting functionality from scratch if an existing implementation can be adapted or extended.
4.  **Iterative Development:**
    -   Write minimal, focused test cases for new features or bug fixes.
    -   Implement the code.
    -   Run tests and iteratively debug until all issues are resolved.

**Testing Guidelines:**
-   **Fixtures:** Use `pytest` fixtures for common test resources to ensure consistency and reduce setup duplication.
-   **Minimalism:** Each test case should be minimal, focusing on a single feature or bug, and have minimal dependencies.
-   **Performance:** Design test cases to be as fast as possible. Employ mocking extensively to isolate units under test and speed up execution.