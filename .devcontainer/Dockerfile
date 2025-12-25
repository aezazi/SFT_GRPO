FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in venv
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA build)
RUN pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace