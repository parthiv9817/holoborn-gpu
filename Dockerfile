# TRELLIS.2 serverless worker — canonical RunPod pattern.
#
# Design: every Python/CUDA dependency is pip-installed against THIS image's
# Python + CUDA. No dist_packages.tar at runtime, no network-volume code.
# Only the 16 GB HuggingFace model cache lives on /runpod-volume.
#
# Build recipe follows TRELLIS.2/setup.sh exactly.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/TRELLIS.2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    HF_HOME=/runpod-volume/trellis_hf_cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    MAX_JOBS=2

# System packages + Python 3.11 via deadsnakes (Ubuntu 22.04 universe ships
# 3.11.0rc1 which has known inspect.getsource bugs that break Triton JIT).
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends software-properties-common gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        git build-essential ca-certificates wget \
        libjpeg-dev libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    wget -qO- https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    printf '#!/bin/bash\nexec "$@"\n' > /usr/local/bin/sudo && \
    chmod +x /usr/local/bin/sudo

# PyTorch 2.6 + torchvision for CUDA 12.4 (matches TRELLIS.2 setup.sh)
RUN pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

# RunPod serverless SDK + HTTP/image libs
RUN pip install --no-cache-dir runpod requests Pillow numpy

# TRELLIS core deps (omit gradio — we don't serve a web UI)
RUN pip install --no-cache-dir \
        imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
        trimesh transformers tensorboard pandas lpips zstandard kornia timm \
        pillow-simd && \
    pip install --no-cache-dir \
        git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Face enhancement chain (--no-deps avoids torch downgrade)
RUN pip install --no-cache-dir --no-deps \
        basicsr==1.4.2 facexlib==0.3.0 realesrgan==0.3.0 gfpgan==1.3.8 && \
    pip install --no-cache-dir \
        addict future lmdb pyyaml scikit-image scipy tb-nightly yapf filterpy numba

# basicsr bundles an import broken by torchvision>=0.16
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
        /usr/local/lib/python3.11/dist-packages/basicsr/data/degradations.py

# Pre-bake enhancement weights (zero network on cold start)
RUN mkdir -p /root/.cache/realesrgan /root/.cache/gfpgan /opt/gfpgan/weights && \
    wget -q -O /root/.cache/realesrgan/RealESRGAN_x2plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth && \
    wget -q -O /opt/gfpgan/weights/GFPGANv1.3.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth && \
    cp /opt/gfpgan/weights/GFPGANv1.3.pth /root/.cache/gfpgan/GFPGANv1.3.pth && \
    wget -q -O /opt/gfpgan/weights/detection_Resnet50_Final.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    wget -q -O /opt/gfpgan/weights/parsing_parsenet.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth

# flash-attn (prebuilt wheel for cu124+py311+torch2.6 — falls back to source build if missing)
RUN pip install --no-cache-dir flash-attn==2.7.3

# nvdiffrast from source (NVlabs; not on PyPI)
RUN git clone --depth 1 -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && \
    pip install --no-cache-dir /tmp/nvdiffrast --no-build-isolation && \
    rm -rf /tmp/nvdiffrast

# cumesh from TRELLIS author's repo (CuBVH + xatlas submodules)
RUN git clone --depth 1 --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh && \
    pip install --no-cache-dir /tmp/CuMesh --no-build-isolation && \
    rm -rf /tmp/CuMesh

# flex_gemm from TRELLIS author's repo
RUN git clone --depth 1 --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM && \
    pip install --no-cache-dir /tmp/FlexGEMM --no-build-isolation && \
    rm -rf /tmp/FlexGEMM

# TRELLIS.2 source (pinned commit that matches HF cache) + DINOv3 patch
RUN git clone https://github.com/microsoft/TRELLIS.2.git /workspace/TRELLIS.2 && \
    cd /workspace/TRELLIS.2 && \
    git checkout 5565d240c4a494caaf9ece7a554542b76ffa36d3 && \
    git submodule update --init --recursive && \
    sed -i 's|for i, layer_module in enumerate(self\.model\.layer):|_layers = self.model.layer if hasattr(self.model, "layer") else self.model.model.layer\n        for i, layer_module in enumerate(_layers):|' \
        trellis2/modules/image_feature_extractor.py && \
    grep -q 'hasattr(self.model, "layer")' trellis2/modules/image_feature_extractor.py && echo "DINOv3 patch applied"

# o-voxel lives inside TRELLIS.2 repo — builds against its own eigen submodule
RUN pip install --no-cache-dir /workspace/TRELLIS.2/o-voxel --no-build-isolation

# Application code
COPY handler.py       /opt/app/handler.py
COPY preprocess.py    /opt/app/preprocess.py
COPY run_inference.py /opt/app/run_inference.py

WORKDIR /workspace

# RunPod serverless canonical pattern: SDK runs its own uvicorn/poll loop
CMD ["python3", "-u", "/opt/app/handler.py"]
