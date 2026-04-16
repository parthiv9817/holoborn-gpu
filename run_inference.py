"""TRELLIS.2 production inference: image -> GLB.

Sequential pipeline matching server.py:
  1. GFPGAN + RealESRGAN x2 enhance
  2. RMBG (BiRefNet) cutout
  3. TRELLIS.2 1536_cascade generation
  4. decode latents -> to_glb (decimate 50k, texture 4k)

Usage:
  python run_inference.py input.jpg output.glb
  python run_inference.py input.jpg output.glb --seed 0 --decimation 50000
"""
import os
import sys
import time
import gc
import types
import argparse

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/workspace/trellis_hf_cache")
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

_TRELLIS_REPO = "/workspace/TRELLIS.2"
if _TRELLIS_REPO not in sys.path:
    sys.path.insert(0, _TRELLIS_REPO)

import torchvision.transforms.functional as _F
_compat = types.ModuleType("torchvision.transforms.functional_tensor")
_compat.rgb_to_grayscale = _F.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _compat

import cv2
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer

from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: E402
import o_voxel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_raw, autocrop_square_from_alpha  # noqa: E402

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


def load_enhancer():
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
    return upsampler, face_restorer


def enhance_image(face_restorer, upsampler, input_path, output_path, *, do_preprocess=True, preprocessed_path=None):
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"cannot decode image at {input_path}")
    ih, iw = img_bgr.shape[:2]
    if do_preprocess:
        before_mean = float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())
        img_bgr, scale = preprocess_raw(img_bgr)
        after_mean = float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())
        nh, nw = img_bgr.shape[:2]
        print(f"[preprocess] CLAHE+WB+upscale: {iw}x{ih} -> {nw}x{nh} "
              f"(scale={scale:.2f}) luma {before_mean:.0f} -> {after_mean:.0f}")
        if preprocessed_path:
            cv2.imwrite(preprocessed_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        ih, iw = nh, nw
    print(f"[enhance] input {iw}x{ih} -> GFPGAN+ESRGAN x2")
    has_face = False
    output = None
    try:
        _, restored_faces, output = face_restorer.enhance(
            img_bgr, has_aligned=False, only_center_face=False, paste_back=True,
        )
        has_face = len(restored_faces) > 0
        print(f"[enhance] GFPGAN detected {len(restored_faces)} face(s)")
    except Exception as e:
        print(f"[enhance] GFPGAN failed: {e!r} -- falling back to ESRGAN only")
        has_face, output = False, None

    if not has_face or output is None:
        print("[enhance] running RealESRGAN fallback (bg only)")
        output, _ = upsampler.enhance(img_bgr, outscale=2)

    cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = output.shape[:2]
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[enhance] done {w}x{h} {size_kb:.0f}KB face={has_face}")
    return has_face


def load_pipeline():
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    return pipeline


def rembg_to_cutout(pipeline, pil_image):
    rgb = pil_image.convert("RGB")
    print(f"[rembg] input {rgb.size} -> GPU")
    pipeline.rembg_model.to("cuda")
    cutout = pipeline.rembg_model(rgb)
    pipeline.rembg_model.cpu()
    torch.cuda.empty_cache()
    print(f"[rembg] done {cutout.size} mode={cutout.mode}")
    return cutout


def generate_glb(pipeline, cutout, output_path, *, seed,
                 decimation_target, texture_size, pipeline_type):
    t0 = time.time()
    outputs = pipeline.run(
        cutout, seed=seed, preprocess_image=True,
        pipeline_type=pipeline_type, return_latent=True,
        **SAMPLER_PARAMS,
    )
    shape_slat, tex_slat, res = outputs[1]
    print(f"[gen] {time.time()-t0:.1f}s, res={res}")

    t0 = time.time()
    fresh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    print(f"[decode] V={fresh.vertices.shape[0]} F={fresh.faces.shape[0]} "
          f"in {time.time()-t0:.1f}s")

    t0 = time.time()
    glb = o_voxel.postprocess.to_glb(
        vertices=fresh.vertices,
        faces=fresh.faces,
        attr_volume=fresh.attrs,
        coords=fresh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True, remesh_band=1, remesh_project=0.5,
        mesh_cluster_threshold_cone_half_angle_rad=np.radians(180.0),
        mesh_cluster_smooth_strength=5,
        mesh_cluster_refine_iterations=2,
        mesh_cluster_global_iterations=2,
    )
    print(f"[glb] postprocess in {time.time()-t0:.1f}s")

    glb.export(output_path)
    sz = os.path.getsize(output_path)
    print(f"[done] {output_path} ({sz/1e6:.2f} MB)")
    return output_path


def main():
    ap = argparse.ArgumentParser(description="TRELLIS.2 image-to-GLB (enhance+rembg+trellis)")
    ap.add_argument("input", help="input image (jpg/png/webp/etc)")
    ap.add_argument("output", help="output GLB path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decimation", type=int, default=50000,
                    help="decimation_target for to_glb")
    ap.add_argument("--texture-size", type=int, default=4096)
    ap.add_argument("--pipeline-type", default="1536_cascade",
                    choices=["512", "1024", "1024_cascade", "1536_cascade"])
    ap.add_argument("--enhanced-path", default="/tmp/enhanced.jpg",
                    help="where to write the GFPGAN/ESRGAN enhanced image")
    ap.add_argument("--save-cutout", default=None,
                    help="optional path to save the RMBG cutout")
    ap.add_argument("--skip-enhance", action="store_true",
                    help="skip GFPGAN+ESRGAN step (rembg+trellis only)")
    ap.add_argument("--skip-preprocess", action="store_true",
                    help="skip CLAHE+WB+upscale before enhance")
    ap.add_argument("--autocrop", action="store_true",
                    help="opt-in: square auto-crop after rembg (TRELLIS already "
                         "does this internally — only use for debugging)")
    ap.add_argument("--preprocessed-path", default="/tmp/preprocessed.jpg",
                    help="where to write the CLAHE+WB+upscale image")
    args = ap.parse_args()

    print(f"[init] torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"dev={torch.cuda.get_device_name(0)}")

    t_start = time.time()

    if args.skip_enhance:
        enhanced_path = args.input
    else:
        t0 = time.time()
        upsampler, face_restorer = load_enhancer()
        print(f"[load] enhancer ready in {time.time()-t0:.1f}s")
        enhance_image(face_restorer, upsampler, args.input, args.enhanced_path,
                      do_preprocess=not args.skip_preprocess,
                      preprocessed_path=args.preprocessed_path)
        enhanced_path = args.enhanced_path
        # free VRAM before TRELLIS, matching server.py
        del upsampler, face_restorer
        gc.collect()
        torch.cuda.empty_cache()
        print("[enhance.unload] freed")

    t0 = time.time()
    pipeline = load_pipeline()
    print(f"[load] pipeline ready in {time.time()-t0:.1f}s")

    cutout = rembg_to_cutout(pipeline, Image.open(enhanced_path))
    if args.autocrop:
        before = cutout.size
        cutout = autocrop_square_from_alpha(cutout)
        print(f"[autocrop] {before} -> {cutout.size} (square, person ~62% fill)")
    if args.save_cutout:
        cutout.save(args.save_cutout)

    generate_glb(
        pipeline, cutout, args.output,
        seed=args.seed,
        decimation_target=args.decimation,
        texture_size=args.texture_size,
        pipeline_type=args.pipeline_type,
    )

    print(f"[total] {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
