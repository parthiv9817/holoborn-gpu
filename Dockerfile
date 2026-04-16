# Avatar-gen server image.
#
# Design: /workspace is a RunPod persistent volume supplying TRELLIS.2/,
# trellis_hf_cache/, and .snapshot/dist_packages.tar. This image bakes
# everything else so a fresh pod boots to a warm server without manual steps.
#
# Build:  docker build -t avatar-gen:latest .
# Run:    docker run --gpus all -p 8000:8000 -v /workspace:/workspace avatar-gen:latest
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

# --- system libs (match RESTORE.sh) ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils \
        git wget curl ca-certificates \
        build-essential libjpeg-dev libgl1 libglib2.0-0 \
        libosmesa6-dev libglu1-mesa && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py && \
    python3.11 /tmp/get-pip.py && rm /tmp/get-pip.py && \
    printf '#!/bin/bash\nexec "$@"\n' > /usr/local/bin/sudo && \
    chmod +x /usr/local/bin/sudo

# --- FastAPI stack (torch + TRELLIS deps come from dist_packages.tar at runtime) ---
RUN pip install --no-cache-dir \
        fastapi uvicorn[standard] python-multipart runpod requests

# --- Enhancement deps, --no-deps to avoid torch downgrade ---
RUN pip install --no-cache-dir --no-deps \
        basicsr==1.4.2 facexlib==0.3.0 realesrgan==0.3.0 gfpgan==1.3.8 && \
    pip install --no-cache-dir \
        opencv-python-headless>=4.11 \
        addict future lmdb pyyaml requests scikit-image scipy tb-nightly tqdm yapf \
        filterpy numba

# --- Pre-bake enhancement weights so first request has zero cold-start ---
RUN mkdir -p /root/.cache/realesrgan /opt/gfpgan/weights && \
    wget -q -O /root/.cache/realesrgan/RealESRGAN_x2plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth && \
    wget -q -O /opt/gfpgan/weights/GFPGANv1.3.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth && \
    wget -q -O /opt/gfpgan/weights/detection_Resnet50_Final.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    wget -q -O /opt/gfpgan/weights/parsing_parsenet.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth

# --- App code ---
COPY server.py        /opt/app/server.py
COPY handler.py       /opt/app/handler.py
COPY preprocess.py    /opt/app/preprocess.py
COPY run_inference.py /opt/app/run_inference.py
COPY entrypoint.sh    /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh

WORKDIR /workspace
EXPOSE 8000

ENTRYPOINT ["/opt/app/entrypoint.sh"]
