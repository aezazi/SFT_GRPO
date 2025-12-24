# Base image: CUDA 12.8 with Ubuntu 24.04
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Avoid interactive prompts during installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        git \
        curl \
        build-essential \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Create a virtual environment at /opt/venv
RUN python -m venv /opt/venv

# Activate venv and upgrade pip
RUN /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

# Add venv to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Copy your project files into the container
COPY . /workspace

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Optional: set default shell to bash
SHELL ["/bin/bash", "-c"]

# Container entrypoint (optional, you can override in devcontainer.json)
CMD ["bash"]