"""
RunPod Serverless handler for avatar-gen.

Wraps the existing FastAPI server.py without modifying its pipeline code.
On worker cold start we boot uvicorn as a subprocess (so RESTORE.sh's
CWD-sensitive TRELLIS init runs exactly as it does on a pod). Once
/health is green we accept jobs and proxy them to localhost:8000/generate.

Input job shape:
    { "image_b64": "<base64 encoded JPEG/PNG>" }
  or
    { "image_url": "https://..." }

Output:
    { "glb_b64": "<base64 GLB>", "glb_size_bytes": int, "elapsed_seconds": float }

Note: RunPod /runsync response cap is ~20 MB. Current GLBs are 25-35 MB,
so /runsync will truncate. Use /run (async) + /status polling, or add an
S3 upload path here before flipping to /runsync for production.
"""
import os
import sys
import time
import base64
import subprocess
import threading

import requests
import runpod

SERVER_PORT = 8000
HEALTH_URL = f"http://127.0.0.1:{SERVER_PORT}/health"
GENERATE_URL = f"http://127.0.0.1:{SERVER_PORT}/generate"
STARTUP_TIMEOUT = 900  # 15 min — RESTORE.sh tar extract + TRELLIS load on cold start

_ready = threading.Event()
_startup_error: list = []


def _spawn_uvicorn() -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "/workspace/TRELLIS.2")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HF_HOME", "/workspace/trellis_hf_cache")
    return subprocess.Popen(
        [
            "python3", "-m", "uvicorn", "server:app",
            "--host", "127.0.0.1",
            "--port", str(SERVER_PORT),
            "--log-level", "info",
        ],
        cwd="/workspace",
        env=env,
    )


def _watch_health(proc: subprocess.Popen) -> None:
    t0 = time.time()
    while time.time() - t0 < STARTUP_TIMEOUT:
        if proc.poll() is not None:
            _startup_error.append(f"uvicorn exited early with code {proc.returncode}")
            return
        try:
            r = requests.get(HEALTH_URL, timeout=2)
            if r.status_code == 200 and r.json().get("trellis_loaded"):
                print(f"[handler] server ready after {time.time() - t0:.1f}s", flush=True)
                _ready.set()
                return
        except Exception:
            pass
        time.sleep(2)
    _startup_error.append(f"server failed to become ready within {STARTUP_TIMEOUT}s")


print("[handler] cold start — spawning uvicorn subprocess", flush=True)
_proc = _spawn_uvicorn()
threading.Thread(target=_watch_health, args=(_proc,), daemon=True).start()


def _fetch_image(inp: dict) -> bytes:
    if "image_b64" in inp:
        return base64.b64decode(inp["image_b64"])
    if "image_url" in inp:
        r = requests.get(inp["image_url"], timeout=30)
        r.raise_for_status()
        return r.content
    raise ValueError("input must contain image_b64 or image_url")


def handler(job):
    t0 = time.time()

    if not _ready.is_set():
        _ready.wait(timeout=STARTUP_TIMEOUT)
    if not _ready.is_set():
        return {"error": "server never became ready", "detail": _startup_error}

    try:
        img_bytes = _fetch_image(job.get("input") or {})
    except Exception as e:
        return {"error": f"bad input: {e}"}

    try:
        resp = requests.post(
            GENERATE_URL,
            files={"image": ("input.jpg", img_bytes, "image/jpeg")},
            timeout=600,
        )
    except Exception as e:
        return {"error": f"pipeline request failed: {e}"}

    if resp.status_code != 200:
        return {
            "error": f"pipeline returned {resp.status_code}",
            "detail": resp.text[:2000],
        }

    glb_bytes = resp.content
    return {
        "glb_b64": base64.b64encode(glb_bytes).decode("ascii"),
        "glb_size_bytes": len(glb_bytes),
        "elapsed_seconds": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
