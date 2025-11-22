# Project Overview

This is a Python project for protein sequence classification. It uses ESM2 models fine-tuned with LoRA adapters. The project is designed to be modular and extensible for both human developers and coding agents.

# Coding best practices:
When coding (i.e. implement new features, fix bugs)
- Ask question to the user to better focus your efforts.
- When key technologies aren't specified, prefer the following:
  - **Machine Learning:** huggingface and pytorch.
  - **Data analysys:** polars over pandas
  - **Testing:** pytest
- Before start coding:
    - Make a plan
    - Gather documentation using context7 MCP and Google Search 
    - Do not repeat yourself: reuse extensively the already implemented code. 
    - You can update existing code if it allows to do not re-write code from scratch.
    - Write a minimal test.
    - Test the code and iteratively solve the bugs.
- Test: 
    - use fixture for common resources
    - each test case must be minimal and address only the needed features 
    - each test case with have minimal dependencies
    - each test case must be designed to be as fast as possible. Use extensively mocking

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
