# Avatar-gen server image (serverless).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/TRELLIS.2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    HF_HOME=/workspace/trellis_hf_cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        git build-essential ca-certificates wget \
        libosmesa6-dev libglu1-mesa libgl1 libglib2.0-0 libjpeg-dev && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    printf '#!/bin/bash\nexec "$@"\n' > /usr/local/bin/sudo && \
    chmod +x /usr/local/bin/sudo

RUN pip install --no-cache-dir \
        fastapi uvicorn[standard] python-multipart \
        runpod requests \
        opencv-python-headless Pillow numpy

RUN mkdir -p /root/.cache/realesrgan /opt/gfpgan/weights && \
    wget -q -O /root/.cache/realesrgan/RealESRGAN_x2plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth && \
    wget -q -O /opt/gfpgan/weights/GFPGANv1.3.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth && \
    wget -q -O /opt/gfpgan/weights/detection_Resnet50_Final.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    wget -q -O /opt/gfpgan/weights/parsing_parsenet.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth

COPY server.py        /opt/app/server.py
COPY preprocess.py    /opt/app/preprocess.py
COPY run_inference.py /opt/app/run_inference.py
COPY handler.py       /opt/app/handler.py
COPY entrypoint.sh    /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh

WORKDIR /workspace
EXPOSE 8000

ENTRYPOINT ["/opt/app/entrypoint.sh"]
