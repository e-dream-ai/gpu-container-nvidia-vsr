# Stage 1: Base image with ComfyUI + Nvidia RTX Nodes
# Uses CUDA 12.4 for VFX SDK compatibility
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libgl1 \
    build-essential \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install comfy-cli
RUN pip install --upgrade pip setuptools wheel
RUN pip install comfy-cli

# Pre-install PyTorch with CUDA 12.4 support
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install \
    --cuda-version 12.4 --nvidia --skip-torch-or-directml

RUN comfy tracking disable

WORKDIR /comfyui

# Install runpod and dependencies
RUN pip install runpod requests websocket-client boto3

# Install Nvidia RTX Nodes (contains RTXVideoSuperResolution node)
# This installs the VFX SDK which supports datacenter GPUs (A100, H100, L40S, A40)
RUN cd custom_nodes && \
    git clone https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.git && \
    cd Nvidia_RTX_Nodes_ComfyUI && \
    pip install -r requirements.txt 2>/dev/null || true && \
    pip install -U --no-build-isolation nvidia-vfx --index-url https://pypi.nvidia.com

# Install ComfyUI-VideoHelperSuite for video I/O (LoadVideo, SaveVideo, CreateVideo, GetVideoComponents)
RUN cd custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt 2>/dev/null || true

RUN pip cache purge

# Support for network volume
ADD src/extra_model_paths.yaml ./

WORKDIR /

# Add scripts
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json test_input_video.json ./
RUN chmod +x /start.sh /restore_snapshot.sh

# Optionally copy snapshot file
ADD *snapshot*.json /

# Restore snapshot for custom nodes
RUN /restore_snapshot.sh

# No models to download — Nvidia VSR uses the VFX SDK, not model files

CMD ["/start.sh"]
