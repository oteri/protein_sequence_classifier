# Project Overview

This is a Python project for protein sequence classification. It uses ESM2 models fine-tuned with LoRA adapters. The project is designed to be modular and extensible for both human developers and coding agents.

## Building and Running

The project is intended to be run using `make`. The following commands are planned:

*   `make train`: This will launch the training process using `accelerate`.

## Development Conventions

The project follows a clear structure:

*   `configs/`: Contains configuration files for training and hardware.
    *   `train.yaml`: For model and training hyperparameters.
    *   `accelerate.yaml`: For hardware and distribution configuration.
*   `data/`: Contains the dataset, split into `train`, `validation`, and `test` sets.
*   `src/`: Contains the main source code.
    *   `esm2_classification.py`: The main script for training and classification.
*   `Makefile`: Contains the commands for running the project.

## Current State

The initial directory structure and files (`configs`, `data`, `src` directories, and `configs/train.yaml`, `configs/accelerate.yaml`, `src/esm2_classification.py`, `Makefile` files) have been created as described in the `README.md` and previous `GEMINI.md` instructions.
