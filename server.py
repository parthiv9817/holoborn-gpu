"""Avatar generation FastAPI server.

POST /generate  — multipart image → GLB
GET  /health    — liveness

Pipeline: enhance (GFPGAN+RealESRGAN) → rembg (via pipeline.rembg_model)
          → TRELLIS.2 1536_cascade → GLB.

NOTE: The working TRELLIS.2 build already outputs head-at-+Y; the historical
Y-flip fix has been intentionally removed (see run_inference.py comment from
2026-04-09). Do not re-apply it — it inverts the mesh.
"""
import os
import sys
import types
import time
import gc
import io
import uuid
import traceback
from datetime import datetime

_C = {
    "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
    "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m",
    "magenta": "\033[35m", "blue": "\033[34m", "red": "\033[31m",
    "grey": "\033[90m",
}

def _vram():
    if not torch.cuda.is_available():
        return "cpu"
    alloc = torch.cuda.memory_allocated() / 1e9
    reserv = torch.cuda.memory_reserved() / 1e9
    return f"vram={alloc:.2f}/{reserv:.2f}GB"

def _log(stage, msg, color="cyan", rid=None):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    tag = f"{_C['dim']}{ts}{_C['reset']}"
    rid_s = f"{_C['grey']}[{rid}]{_C['reset']} " if rid else ""
    head = f"{_C['bold']}{_C[color]}[{stage}]{_C['reset']}"
    print(f"{tag} {rid_s}{head} {msg}", flush=True)

def _banner(text, color="magenta"):
    line = "=" * 72
    print(f"{_C['bold']}{_C[color]}{line}\n{text}\n{line}{_C['reset']}", flush=True)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("HF_HOME", "/workspace/trellis_hf_cache")
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
# os.environ.setdefault("HF_HUB_OFFLINE", "1")
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

os.chdir("/workspace")
sys.path.insert(0, "/workspace/TRELLIS.2")

import torchvision.transforms.functional as _F
_compat = types.ModuleType("torchvision.transforms.functional_tensor")
_compat.rgb_to_grayscale = _F.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _compat

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import uvicorn

from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

sys.path.insert(0, "/workspace")
from preprocess import preprocess_raw  # noqa: E402


SAMPLER_PARAMS = dict(
    sparse_structure_sampler_params={
        "steps": 12, "guidance_strength": 8.0,
        "guidance_rescale": 0.7, "rescale_t": 5.0,
    },
    shape_slat_sampler_params={
        "steps": 12, "guidance_strength": 8.0,
        "guidance_rescale": 0.5, "rescale_t": 3.0,
    },
    tex_slat_sampler_params={
        "steps": 12, "guidance_strength": 1.0,
        "guidance_rescale": 0.0, "rescale_t": 3.0,
    },
)

app = FastAPI(title="avatar-gen")

upsampler = None
face_restorer = None
trellis_pipeline = None


def load_enhancer(rid=None):
    global upsampler, face_restorer
    if face_restorer is not None:
        _log("enhance.load", f"cached ({_vram()})", "green", rid)
        return
    _log("enhance.load", f"loading RealESRGAN+GFPGAN... ({_vram()})", "yellow", rid)
    t0 = time.time()
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        model=model, tile=0, tile_pad=10, pre_pad=0,
        half=True, device="cuda",
    )
    face_restorer = GFPGANer(
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        upscale=2, arch="clean", channel_multiplier=2,
        bg_upsampler=upsampler, device="cuda",
    )
    _log("enhance.load", f"ready in {time.time()-t0:.2f}s ({_vram()})", "green", rid)


def unload_enhancer(rid=None):
    global upsampler, face_restorer
    upsampler = None
    face_restorer = None
    gc.collect()
    torch.cuda.empty_cache()
    _log("enhance.unload", f"freed ({_vram()})", "grey", rid)


def load_trellis(rid=None):
    global trellis_pipeline
    if trellis_pipeline is not None:
        return
    _log("trellis.load", f"loading TRELLIS.2-4B pipeline... ({_vram()})", "yellow", rid)
    t0 = time.time()
    p = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    p.cuda()
    trellis_pipeline = p
    _log("trellis.load", f"ready in {time.time()-t0:.2f}s ({_vram()})", "green", rid)


def enhance_image(input_path: str, output_path: str, rid=None) -> bool:
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, f"cannot decode image at {input_path}")
    ih, iw = img_bgr.shape[:2]

    before_luma = float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())
    img_bgr, scale = preprocess_raw(img_bgr)
    after_luma = float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())
    nh, nw = img_bgr.shape[:2]
    _log("preprocess", f"CLAHE+WB+upscale: {iw}x{ih} → {nw}x{nh} "
         f"(scale={scale:.2f}) luma {before_luma:.0f}→{after_luma:.0f}",
         "cyan", rid)
    ih, iw = nh, nw

    _log("enhance", f"input {iw}x{ih} → GFPGAN+ESRGAN x2 ({_vram()})", "cyan", rid)
    t0 = time.time()
    has_face = False
    output = None
    try:
        _, restored_faces, output = face_restorer.enhance(
            img_bgr, has_aligned=False, only_center_face=False, paste_back=True,
        )
        has_face = len(restored_faces) > 0
        _log("enhance", f"GFPGAN detected {len(restored_faces)} face(s)",
             "green" if has_face else "yellow", rid)
    except Exception as e:
        _log("enhance", f"GFPGAN failed: {e!r} — falling back to ESRGAN only",
             "red", rid)
        has_face, output = False, None

    if not has_face or output is None:
        _log("enhance", "running RealESRGAN fallback (bg only)", "yellow", rid)
        output, _ = upsampler.enhance(img_bgr, outscale=2)

    cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = output.shape[:2]
    size_kb = os.path.getsize(output_path) / 1024
    _log("enhance", f"done {w}x{h} {size_kb:.0f}KB face={has_face} "
         f"in {time.time()-t0:.2f}s", "green", rid)
    return has_face


def rembg_to_cutout(pil_image: Image.Image, rid=None) -> Image.Image:
    t0 = time.time()
    rgb = pil_image.convert("RGB")
    _log("rembg", f"input {rgb.size} → GPU ({_vram()})", "cyan", rid)
    trellis_pipeline.rembg_model.to("cuda")
    cutout = trellis_pipeline.rembg_model(rgb)
    trellis_pipeline.rembg_model.cpu()
    torch.cuda.empty_cache()
    _log("rembg", f"done {cutout.size} mode={cutout.mode} "
         f"in {time.time()-t0:.2f}s ({_vram()})", "green", rid)
    return cutout


def generate_glb(cutout: Image.Image, output_glb_path: str, rid=None):
    _log("trellis.run", f"start (1536_cascade, seed=42, cutout={cutout.size}) "
         f"({_vram()})", "cyan", rid)
    t0 = time.time()
    outputs = trellis_pipeline.run(
        cutout, seed=42, preprocess_image=True,
        pipeline_type="1536_cascade", return_latent=True,
        **SAMPLER_PARAMS,
    )
    shape_slat, tex_slat, res = outputs[1]
    _log("trellis.run", f"done res={res} in {time.time()-t0:.2f}s ({_vram()})",
         "green", rid)

    t0 = time.time()
    _log("trellis.decode", f"decoding shape+tex latents (res={res})", "cyan", rid)
    fresh = trellis_pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    _log("trellis.decode", f"V={fresh.vertices.shape[0]} F={fresh.faces.shape[0]} "
         f"in {time.time()-t0:.2f}s ({_vram()})", "green", rid)

    t0 = time.time()
    _log("postprocess", "to_glb: remesh+decimate(50k)+bake(4k tex)", "cyan", rid)
    glb = o_voxel.postprocess.to_glb(
        vertices=fresh.vertices,
        faces=fresh.faces,
        attr_volume=fresh.attrs,
        coords=fresh.coords,
        attr_layout=trellis_pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=50000,
        texture_size=4096,
        remesh=True, remesh_band=1, remesh_project=0.5,
        mesh_cluster_threshold_cone_half_angle_rad=np.radians(180.0),
        mesh_cluster_smooth_strength=5,
        mesh_cluster_refine_iterations=2,
        mesh_cluster_global_iterations=2,
    )
    glb.export(output_glb_path)
    size_mb = os.path.getsize(output_glb_path) / 1e6
    _log("postprocess", f"glb written {output_glb_path} {size_mb:.2f}MB "
         f"in {time.time()-t0:.2f}s", "green", rid)


@app.on_event("startup")
def _startup():
    _banner("  avatar-gen server booting", "blue")
    _log("boot", f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
         f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
         "cyan")
    load_trellis()
    _banner("  READY — POST /generate (multipart: image)", "green")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "trellis_loaded": trellis_pipeline is not None,
        "enhancer_loaded": face_restorer is not None,
    }


@app.post("/generate")
async def generate(image: UploadFile = File(...)):
    rid = uuid.uuid4().hex[:8]
    t_start = time.time()
    _banner(f"  ▶  /generate  rid={rid}  file={image.filename}", "magenta")
    try:
        input_path = "/tmp/input.jpg"
        enhanced_path = "/tmp/enhanced.jpg"
        output_path = "/tmp/output.glb"

        data = await image.read()
        with open(input_path, "wb") as f:
            f.write(data)
        _log("recv", f"{image.filename} {len(data)/1024:.1f}KB → {input_path}",
             "cyan", rid)

        load_enhancer(rid)
        enhance_image(input_path, enhanced_path, rid)
        unload_enhancer(rid)

        load_trellis(rid)
        cutout = rembg_to_cutout(Image.open(enhanced_path), rid)
        generate_glb(cutout, output_path, rid)

        total = time.time() - t_start
        _banner(f"  ✔  rid={rid}  total={total:.2f}s  → {output_path}", "green")
        return FileResponse(
            output_path, media_type="application/octet-stream",
            filename="avatar.glb",
        )
    except HTTPException:
        raise
    except Exception as e:
        _log("error", f"{e!r}", "red", rid)
        traceback.print_exc()
        _banner(f"  ✘  rid={rid}  FAILED after {time.time()-t_start:.2f}s", "red")
        raise HTTPException(500, f"pipeline error: {e!r}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
