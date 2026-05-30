FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    google-perftools \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install av requests runpod boto3 \
    && pip install -U --no-build-isolation nvidia-vfx --index-url https://pypi.nvidia.com

RUN pip cache purge

ADD src/start.sh src/rp_handler.py test_input_video.json /
RUN chmod +x /start.sh

CMD ["/start.sh"]
