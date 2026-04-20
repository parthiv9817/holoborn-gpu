#!/bin/bash
set -euo pipefail

PY_VER=3.11
DIST_DIR=/usr/local/lib/python${PY_VER}/dist-packages

log() { printf '\033[1;32m[entry]\033[0m %s\n' "$*"; }

# --- 0. RunPod serverless mounts network volumes at /runpod-volume, not /workspace.
#        Symlink so all downstream paths resolve. ---
if [[ -d /runpod-volume/.snapshot ]]; then
    log "network volume detected at /runpod-volume — linking to /workspace"
    # /workspace exists as the WORKDIR from Dockerfile but is empty on serverless
    rm -rf /workspace
    ln -sf /runpod-volume /workspace
    # Point PYTHONPATH at the real path rather than the symlink. Cheap insurance
    # against any import-path-resolution oddities with lazy __getattr__ loaders.
    export PYTHONPATH=/runpod-volume/TRELLIS.2
fi

# Mirror stdout+stderr to a file on the volume so the full boot log survives
# container death. Retrievable from outside via the volume's S3 endpoint.
# RunPod's Container Logs UI is not always accessible; this is the fallback.
if touch /workspace/.boot.log 2>/dev/null; then
    : > /workspace/.boot.log  # truncate on each boot
    exec > >(tee -a /workspace/.boot.log) 2>&1
    log "boot log mirrored to /workspace/.boot.log (also stdout)"
fi

if [[ ! -f /workspace/.snapshot/dist_packages.tar ]]; then
    echo "ERROR: /workspace/.snapshot/dist_packages.tar not found."
    echo "Mount the persistent pod volume at /workspace (must contain TRELLIS.2/, trellis_hf_cache/, .snapshot/)."
    exit 1
fi

log "restoring dist-packages from tar (~7 GB)..."
mkdir -p "$(dirname ${DIST_DIR})"
tar -xf /workspace/.snapshot/dist_packages.tar -C /usr/local/lib/python${PY_VER} --overwrite
log "dist-packages restored"

DEGRADATIONS="${DIST_DIR}/basicsr/data/degradations.py"
if [[ -f "$DEGRADATIONS" ]]; then
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$DEGRADATIONS" 2>/dev/null || true
fi

log "linking HF cache..."
mkdir -p /root/.cache
rm -rf /root/.cache/huggingface 2>/dev/null || true
ln -s /workspace/trellis_hf_cache /root/.cache/huggingface

HUB=/workspace/trellis_hf_cache/hub
if [[ -d "${HUB}/models--1038lab--RMBG-2.0" ]] && [[ ! -d "${HUB}/models--briaai--RMBG-2.0" ]]; then
    ln -s models--1038lab--RMBG-2.0 "${HUB}/models--briaai--RMBG-2.0"
fi
if [[ -d "${HUB}/models--camenduru--dinov3-vitl16-pretrain-lvd1689m" ]] && [[ ! -d "${HUB}/models--facebook--dinov3-vitl16-pretrain-lvd1689m" ]]; then
    ln -s models--camenduru--dinov3-vitl16-pretrain-lvd1689m "${HUB}/models--facebook--dinov3-vitl16-pretrain-lvd1689m"
fi
log "HF cache linked"

# Nuke any __pycache__ that leaked onto the volume. Stale .pyc compiled from an
# older version of the code would make modules look like the wrong shape
# (e.g., "cannot import name 'Mesh' from trellis2.representations").
# PYTHONDONTWRITEBYTECODE=1 keeps them from being recreated.
log "clearing stale __pycache__ under /workspace/TRELLIS.2/..."
find /workspace/TRELLIS.2 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

PATCH_FILE=/workspace/TRELLIS.2/trellis2/modules/image_feature_extractor.py
if [[ -f "$PATCH_FILE" ]] && ! grep -q 'hasattr(self.model, "layer")' "$PATCH_FILE" 2>/dev/null; then
    log "applying DINOv3 layer-attribute patch..."
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

log "seeding enhancement weights..."
mkdir -p /workspace/gfpgan/weights
cp -n /opt/gfpgan/weights/detection_Resnet50_Final.pth /workspace/gfpgan/weights/ 2>/dev/null || true
cp -n /opt/gfpgan/weights/parsing_parsenet.pth /workspace/gfpgan/weights/ 2>/dev/null || true
cp -n /opt/gfpgan/weights/GFPGANv1.3.pth /workspace/gfpgan/weights/ 2>/dev/null || true
mkdir -p /root/.cache/realesrgan
mkdir -p /root/.cache/gfpgan
cp -n /opt/gfpgan/weights/GFPGANv1.3.pth /root/.cache/gfpgan/GFPGANv1.3.pth 2>/dev/null || true

log "seeding app code..."
cd /workspace
for f in server.py preprocess.py run_inference.py; do
    if [[ ! -f /workspace/$f ]]; then
        log "no /workspace/$f — using baked copy"
        cp /opt/app/$f /workspace/$f
    fi
done
cp /opt/app/handler.py /workspace/handler.py

log "running pre-handler import diagnostic..."
python3 -c "
import sys
sys.path.insert(0, '/runpod-volume/TRELLIS.2')
from trellis2.representations import Mesh, MeshWithVoxel
print('[diag] Mesh import OK:', Mesh)
from trellis2.pipelines import Trellis2ImageTo3DPipeline
print('[diag] Pipeline import OK:', Trellis2ImageTo3DPipeline)
" || echo '[diag] IMPORT FAILED — see traceback above'

log "starting RunPod serverless handler..."
exec python3 /workspace/handler.py
