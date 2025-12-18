# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set the working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    make \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project configuration files
COPY pyproject.toml uv.lock Makefile ./

# Install dependencies into the system site-packages
# This leverages the pre-installed PyTorch in the base image
RUN uv pip install --system .

# Copy the rest of the application
COPY src ./src
COPY configs ./configs
COPY data ./data

# Set the entrypoint to run the training command
# We need to adjust Makefile or call accelerate directly because we are in system python
ENV ACCELERATE_CONFIG=configs/accelerate_runpod.yaml
CMD ["make", "train"]
