#!/bin/bash
# RESTORE.sh — bring a fresh/restarted RunPod pod back to full working state.
# Restores: TRELLIS.2 (optimized params)
#
# Run this ONCE on a fresh pod after stop/restart:
#   bash /workspace/.snapshot/RESTORE.sh
#
# What's on /workspace (persists across restarts):
#   - TRELLIS.2/              TRELLIS code
#   - trellis_hf_cache/       TRELLIS HF model weights
#   - miniconda3/             Miniconda (base for future conda envs like DECA/FLAME)
#   - .snapshot/              This script + dist_packages.tar
#   - run_inference.py        TRELLIS inference with optimized params
#   - test5-9.jpg/glb         Test images and outputs
#
# What gets wiped on restart (root overlay) and needs restoring:
#   - /usr/local/lib/python3.11/dist-packages/  (TRELLIS python packages)
#   - /root/.cache/huggingface/                  (symlink to workspace)
#   - System apt packages
#   - pip-installed packages (pyrender, GFPGAN, etc.)
#
# Updated: 2026-04-10 (ACE++ removed, optimized TRELLIS params)

set -euo pipefail

SNAP=/workspace/.snapshot
PY_VER=3.11
DIST_DIR=/usr/local/lib/python${PY_VER}/dist-packages

log() { printf '\033[1;36m[restore]\033[0m %s\n' "$*"; }

# -------------------------------------------------------------------
# Step 1: Check environment
# -------------------------------------------------------------------
log "checking environment..."
. /etc/os-release
if [[ "${VERSION_ID:-}" != "22.04" ]]; then
    echo "WARNING: OS version is $VERSION_ID, expected 22.04."
fi
if ! command -v python${PY_VER} >/dev/null; then
    echo "ERROR: python${PY_VER} not found."
    exit 1
fi
log "Ubuntu ${VERSION_ID} + Python ${PY_VER} ✓"

# -------------------------------------------------------------------
# Step 2: apt deps
# -------------------------------------------------------------------
log "[1/7] installing system libs..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    git build-essential libjpeg-dev libgl1 libglib2.0-0 ca-certificates \
    libosmesa6-dev libglu1-mesa \
    > /dev/null
log "apt deps ✓"

# -------------------------------------------------------------------
# Step 3: sudo stub
# -------------------------------------------------------------------
if [[ ! -x /usr/local/bin/sudo ]]; then
    log "[2/7] installing sudo stub..."
    printf '#!/bin/bash\nexec "$@"\n' > /usr/local/bin/sudo
    chmod +x /usr/local/bin/sudo
else
    log "[2/7] sudo stub already present ✓"
fi

# -------------------------------------------------------------------
# Step 4: restore TRELLIS dist-packages
# -------------------------------------------------------------------
log "[3/7] restoring dist-packages from tar (~7 GB)..."
if [[ ! -f "${SNAP}/dist_packages.tar" ]]; then
    echo "ERROR: ${SNAP}/dist_packages.tar not found"
    exit 1
fi
mkdir -p "$(dirname ${DIST_DIR})"
tar -xf "${SNAP}/dist_packages.tar" -C /usr/local/lib/python${PY_VER}
log "dist-packages restored ✓"

# -------------------------------------------------------------------
# Step 5: Install additional pip packages (wiped on restart)
# -------------------------------------------------------------------
log "[4/7] installing additional pip packages..."
pip install pyrender 'pyglet<2' 'PyOpenGL>=3.1.7' --cache-dir /workspace/pip_cache -q 2>/dev/null
pip install --no-deps basicsr==1.4.2 facexlib==0.3.0 realesrgan==0.3.0 gfpgan==1.3.8 --cache-dir /workspace/pip_cache -q 2>/dev/null
pip install opencv-python-headless addict future lmdb pyyaml scikit-image scipy tb-nightly tqdm yapf filterpy numba --cache-dir /workspace/pip_cache -q 2>/dev/null
# Download GFPGANv1.3.pth if missing
if [ ! -f /workspace/gfpgan/weights/GFPGANv1.3.pth ]; then
    log "downloading GFPGANv1.3.pth..."
    mkdir -p /workspace/gfpgan/weights
    wget -q -O /workspace/gfpgan/weights/GFPGANv1.3.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
fi
# Fix basicsr torchvision compat if needed
DEGRADATIONS="/usr/local/lib/python${PY_VER}/dist-packages/basicsr/data/degradations.py"
if [[ -f "$DEGRADATIONS" ]]; then
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$DEGRADATIONS" 2>/dev/null || true
fi
log "pip packages ✓"

# -------------------------------------------------------------------
# Step 6: symlink TRELLIS HF cache from workspace
# -------------------------------------------------------------------
log "[5/7] linking TRELLIS HF cache..."
mkdir -p /root/.cache
if [[ -L /root/.cache/huggingface ]]; then
    rm /root/.cache/huggingface
fi
if [[ -d /root/.cache/huggingface ]]; then
    rm -rf /root/.cache/huggingface
fi
ln -s /workspace/trellis_hf_cache /root/.cache/huggingface

HUB=/workspace/trellis_hf_cache/hub
if [[ -d "${HUB}/models--1038lab--RMBG-2.0" ]] && [[ ! -d "${HUB}/models--briaai--RMBG-2.0" ]]; then
    ln -s models--1038lab--RMBG-2.0 "${HUB}/models--briaai--RMBG-2.0"
    log "  symlinked briaai/RMBG-2.0 -> 1038lab/RMBG-2.0"
fi
if [[ -d "${HUB}/models--camenduru--dinov3-vitl16-pretrain-lvd1689m" ]] && [[ ! -d "${HUB}/models--facebook--dinov3-vitl16-pretrain-lvd1689m" ]]; then
    ln -s models--camenduru--dinov3-vitl16-pretrain-lvd1689m "${HUB}/models--facebook--dinov3-vitl16-pretrain-lvd1689m"
    log "  symlinked facebook/dinov3 -> camenduru/dinov3"
fi
log "HF cache linked ✓"

# -------------------------------------------------------------------
# Step 7: DINOv3 layer patch (idempotent)
# -------------------------------------------------------------------
PATCH_FILE=/workspace/TRELLIS.2/trellis2/modules/image_feature_extractor.py
if grep -q 'hasattr(self.model, "layer")' "${PATCH_FILE}" 2>/dev/null; then
    log "[6/7] DINOv3 layer patch already present ✓"
else
    log "[6/7] applying DINOv3 layer-attribute patch..."
    python3 - <<'PYEOF'
p = '/workspace/TRELLIS.2/trellis2/modules/image_feature_extractor.py'
s = open(p).read()
old = 'for i, layer_module in enumerate(self.model.layer):'
new = ('_layers = self.model.layer if hasattr(self.model, "layer") '
       'else self.model.model.layer\n'
       '        for i, layer_module in enumerate(_layers):')
if old in s:
    open(p, 'w').write(s.replace(old, new))
    print('patched')
else:
    print('patch target not found — already patched or source changed')
PYEOF
fi

# -------------------------------------------------------------------
# Step 8: sanity checks
# -------------------------------------------------------------------
log "[7/7] running sanity checks..."

export PYTHONPATH=/workspace/TRELLIS.2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENCV_IO_ENABLE_OPENEXR=1
python3 - <<'PYEOF'
import torch
print('  torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  device:', torch.cuda.get_device_name(0))
import flash_attn
print('  flash_attn:', flash_attn.__version__)
import nvdiffrast, cumesh, flex_gemm, o_voxel
print('  cuda exts: ok')
import sys
sys.path.insert(0, '/workspace/TRELLIS.2')
from trellis2.pipelines import Trellis2ImageTo3DPipeline
print('  trellis2: ok')
PYEOF
log "TRELLIS ✓"

# Check miniconda is healthy
if [[ -x /workspace/miniconda3/bin/conda ]]; then
    /workspace/miniconda3/bin/conda --version 2>/dev/null && log "miniconda ✓" || log "miniconda: broken, reinstall with Miniconda3-latest"
else
    log "miniconda: not found (install fresh if needed for DECA/FLAME)"
fi

cat <<'EOF'

RESTORE COMPLETE — TRELLIS.2 (optimized)

Environment variables for TRELLIS (system python):
    export PYTHONPATH=/workspace/TRELLIS.2
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OPENCV_IO_ENABLE_OPENEXR=1
    export HF_HOME=/workspace/trellis_hf_cache
    export HF_TOKEN=<your_token>

Run TRELLIS inference (optimized params):
    python3 /workspace/run_inference.py input.jpg output.glb

Miniconda (for DECA/FLAME env):
    /workspace/miniconda3/bin/conda create -n face3d python=3.10 -y

Test images: test5.jpg through test9.jpg
Test GLBs:   test7.glb, test8.glb, test9.glb (optimized params)
EOF
