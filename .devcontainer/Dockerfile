## Start with Ubuntu 24.04
FROM ubuntu:24.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    nano \
    htop \
    tmux \
    ca-certificates \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install NVIDIA CUDA Toolkit 12.8 (matching your Lambda Labs setup)
# Using NVIDIA's official repositories
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-8 && \
    rm cuda-keyring_1.1-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV PATH=/usr/local/cuda-12.8/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda-12.8

# Install PyTorch 2.9.1 with CUDA 12.8 support
RUN pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# Install essential build dependencies for compiling extensions (like flash-attn)
RUN apt-get update && apt-get install -y \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global user.email "aezazi@icloud.com" && \
    git config --global user.name "aezazi"
    
# Create workspace directory
WORKDIR /workspace/SFT_GRPO

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Default command
CMD ["/bin/bash"]