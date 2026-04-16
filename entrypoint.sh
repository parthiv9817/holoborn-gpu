#!/bin/bash
# entrypoint: restore TRELLIS from /workspace volume, seed weights, start uvicorn.
set -euo pipefail

log() { printf '\033[1;32m[entry]\033[0m %s\n' "$*"; }

# --- 1. sanity-check the mounted volume ---
if [[ ! -f /workspace/.snapshot/dist_packages.tar ]]; then
    echo "ERROR: /workspace/.snapshot/dist_packages.tar not found."
    echo "Mount the persistent pod volume at /workspace (must contain TRELLIS.2/, trellis_hf_cache/, .snapshot/)."
    exit 1
fi

# --- 2. restore TRELLIS python packages + HF cache + DINOv3 patch ---
# RESTORE.sh is idempotent and handles apt / dist-packages / HF symlinks / patch
log "running RESTORE.sh (idempotent)..."
bash /workspace/.snapshot/RESTORE.sh

# --- 3. seed pre-baked enhancement weights into the CWD-relative paths
#        that facexlib + gfpgan look at (they use os.getcwd()). ---
log "seeding enhancement weights..."
mkdir -p /workspace/gfpgan/weights
cp -n /opt/gfpgan/weights/detection_Resnet50_Final.pth /workspace/gfpgan/weights/
cp -n /opt/gfpgan/weights/parsing_parsenet.pth         /workspace/gfpgan/weights/
cp -n /opt/gfpgan/weights/GFPGANv1.3.pth               /workspace/gfpgan/weights/ 2>/dev/null || true
mkdir -p /root/.cache/realesrgan
cp -n /opt/gfpgan/weights/GFPGANv1.3.pth /root/.cache/gfpgan/GFPGANv1.3.pth 2>/dev/null || \
    (mkdir -p /root/.cache/gfpgan && cp /opt/gfpgan/weights/GFPGANv1.3.pth /root/.cache/gfpgan/)

# --- 4. start the FastAPI server from /workspace so CWD-relative paths resolve ---
log "starting uvicorn on :8000 ..."
cd /workspace
# Seed app code onto the volume if missing — the volume copy is authoritative
# once present, so live edits persist across container restarts.
for f in server.py preprocess.py run_inference.py; do
    if [[ ! -f /workspace/$f ]]; then
        log "no /workspace/$f — using baked copy"
        cp /opt/app/$f /workspace/$f
    fi
done
exec python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --log-level info
