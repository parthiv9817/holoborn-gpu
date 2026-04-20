"""RunPod Serverless handler for TRELLIS.2 avatar-gen.

Canonical RunPod pattern: the pipeline + enhancer are loaded at module
import (so they stay resident between jobs), and `runpod.serverless.start`
drives the worker. No FastAPI subprocess, no HTTP proxy — the handler
calls run_inference's functions directly.

Input:
    { "image_b64": "<base64 JPEG/PNG>" }       or
    { "image_url": "https://..." }
Optional keys:
    seed, decimation, texture_size, pipeline_type,
    skip_enhance (bool), skip_preprocess (bool)

Output:
    { "glb_b64": "<base64>",
      "glb_size_bytes": int,
      "elapsed_seconds": float }
"""
import base64
import gc
import os
import sys
import tempfile
import time
import traceback

# Env MUST be set before any TRELLIS / HF imports happen below.
os.environ.setdefault("HF_HOME", "/runpod-volume/trellis_hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# TRELLIS.2 lives at /workspace/TRELLIS.2 (baked into the image by Dockerfile).
sys.path.insert(0, "/workspace/TRELLIS.2")
sys.path.insert(0, "/opt/app")

import requests
import runpod
import torch
from PIL import Image

_t0 = time.time()
print("[handler] cold start: loading TRELLIS pipeline...", flush=True)

import run_inference as ri

# Load the generation pipeline once. Kept resident across jobs.
# Enhancer is loaded/freed per-request (2+ GB VRAM; TRELLIS needs the headroom).
_pipeline = ri.load_pipeline()
print(f"[handler] pipeline ready in {time.time() - _t0:.1f}s", flush=True)


def _fetch_image(inp: dict) -> bytes:
    if "image_b64" in inp:
        return base64.b64decode(inp["image_b64"])
    if "image_url" in inp:
        r = requests.get(inp["image_url"], timeout=30)
        r.raise_for_status()
        return r.content
    raise ValueError("input must contain image_b64 or image_url")


def handler(job):
    t_start = time.time()
    inp = job.get("input") or {}
    seed = int(inp.get("seed", 42))
    decimation = int(inp.get("decimation", 50000))
    texture_size = int(inp.get("texture_size", 4096))
    pipeline_type = inp.get("pipeline_type", "1536_cascade")
    skip_enhance = bool(inp.get("skip_enhance", False))
    skip_preprocess = bool(inp.get("skip_preprocess", False))

    try:
        img_bytes = _fetch_image(inp)
    except Exception as e:
        return {"error": f"bad input: {e}"}

    try:
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "input.jpg")
            enh_path = os.path.join(td, "enhanced.jpg")
            out_path = os.path.join(td, "out.glb")
            with open(in_path, "wb") as f:
                f.write(img_bytes)

            if skip_enhance:
                enhanced_path = in_path
            else:
                upsampler, face_restorer = ri.load_enhancer()
                ri.enhance_image(
                    face_restorer, upsampler, in_path, enh_path,
                    do_preprocess=not skip_preprocess,
                )
                enhanced_path = enh_path
                del upsampler, face_restorer
                gc.collect()
                torch.cuda.empty_cache()

            cutout = ri.rembg_to_cutout(_pipeline, Image.open(enhanced_path))
            ri.generate_glb(
                _pipeline, cutout, out_path,
                seed=seed,
                decimation_target=decimation,
                texture_size=texture_size,
                pipeline_type=pipeline_type,
            )

            with open(out_path, "rb") as f:
                glb_bytes = f.read()
    except Exception as e:
        traceback.print_exc()
        return {"error": f"pipeline failed: {e}"}

    return {
        "glb_b64": base64.b64encode(glb_bytes).decode("ascii"),
        "glb_size_bytes": len(glb_bytes),
        "elapsed_seconds": round(time.time() - t_start, 2),
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
