"""RunPod Serverless handler — SkyReels-V2 Video Generation Engine (Phase 1).

Zero external storage: every response returns the .mp4 as a Base64 string.

Dispatch is by `input.operation`:
  - "image_to_video"     -> animate a reference/keyframe image into a scene
  - "scene_continuation" -> continue from the LAST FRAME of the previous clip
  - "text_to_video"      -> atmospheric/fallback generation (no reference image)

Multi-character scenes: pass a COMPOSITE CANVAS image (Character A left, B right)
as `image_base64` and use spatial language in the prompt
("the character on the left in the blue coat ...").

Models are warm-loaded into VRAM at container startup (see PRELOAD/*_MODEL env
vars), NOT per request. A single-slot cache swaps models only when a request
needs a different one, to respect single-GPU VRAM.
"""

from __future__ import annotations

import base64
import gc
import io
import os
import random
import tempfile
import time
import traceback

import imageio
import torch
from PIL import Image

# torch.compile/inductor tries to gcc-compile CUDA utils linking -lcuda, which
# fails on RunPod workers where libcuda isn't linkable ->
# "BackendCompilerFailed: gcc ... -lcuda ... exit status 1". Fall back to eager
# so inference never dies on a compile issue.
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import runpod

from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import (
    Image2VideoPipeline,
    Text2VideoPipeline,
    resizecrop,
)

# Diffusion-forcing pipeline is optional (used for stronger continuation).
try:
    from skyreels_v2_infer.pipelines import DiffusionForcingPipeline  # type: ignore
    _HAS_DF = True
except Exception:  # pragma: no cover - depends on repo version
    DiffusionForcingPipeline = None  # type: ignore
    _HAS_DF = False

# ----------------------------- configuration --------------------------------

RESOLUTION = os.environ.get("RESOLUTION", "540P").upper()  # 540P | 720P
I2V_MODEL = os.environ.get("I2V_MODEL", "Skywork/SkyReels-V2-I2V-1.3B-540P")
T2V_MODEL = os.environ.get("T2V_MODEL", "Skywork/SkyReels-V2-T2V-14B-540P")
DF_MODEL = os.environ.get("DF_MODEL", "Skywork/SkyReels-V2-DF-1.3B-540P")
# Which pipeline to warm-load at startup: i2v (default), t2v, df, or none.
PRELOAD = os.environ.get("PRELOAD", "i2v").lower()
OFFLOAD = os.environ.get("OFFLOAD", "1") == "1"        # CPU-offload to save VRAM
ALLOW_MULTI_MODEL = os.environ.get("ALLOW_MULTI_MODEL", "0") == "1"

DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def _dims(resolution: str) -> tuple[int, int]:
    if resolution == "540P":
        return 544, 960          # (height, width)
    if resolution == "720P":
        return 720, 1280
    raise ValueError(f"Invalid RESOLUTION: {resolution!r} (use 540P or 720P)")


# ----------------------------- model cache ----------------------------------
# Single-slot cache: {"key": (kind, model_id), "pipe": pipeline}
_CACHE: dict[str, object] = {"key": None, "pipe": None}


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_pipeline(kind: str, model_id: str):
    """Instantiate a SkyReels pipeline; download weights if missing."""
    resolved = download_model(model_id)
    if kind == "i2v":
        return Image2VideoPipeline(model_path=resolved, dit_path=resolved,
                                   use_usp=False, offload=OFFLOAD)
    if kind == "t2v":
        return Text2VideoPipeline(model_path=resolved, dit_path=resolved,
                                  use_usp=False, offload=OFFLOAD)
    if kind == "df":
        if not _HAS_DF:
            raise RuntimeError("DiffusionForcingPipeline not available in this build")
        return DiffusionForcingPipeline(model_path=resolved, dit_path=resolved,
                                        use_usp=False, offload=OFFLOAD)
    raise ValueError(f"unknown pipeline kind: {kind}")


def get_pipeline(kind: str, model_id: str):
    """Return a warm pipeline, loading/swapping as needed (single-slot by default)."""
    key = (kind, model_id)
    if _CACHE["key"] == key and _CACHE["pipe"] is not None:
        return _CACHE["pipe"]
    if not ALLOW_MULTI_MODEL and _CACHE["pipe"] is not None:
        # Evict the previous model to free VRAM before loading a new one.
        _CACHE["pipe"] = None
        _CACHE["key"] = None
        _free_gpu()
    pipe = _load_pipeline(kind, model_id)
    _CACHE["key"] = key
    _CACHE["pipe"] = pipe
    return pipe


# ----------------------------- io helpers -----------------------------------

def _b64_to_pil(b64: str) -> Image.Image:
    if not b64:
        raise ValueError("empty image_base64")
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]         # strip data-URI prefix if present
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _frames_to_mp4_b64(frames, fps: int) -> str:
    """Encode a list of frames to an H.264 mp4 and return Base64 (no disk kept)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        imageio.mimwrite(tmp.name, frames, fps=fps, quality=8,
                         output_params=["-loglevel", "error"])
        with open(tmp.name, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _resolve_seed(seed) -> int:
    if seed is None:
        random.seed(time.time())
        return int(random.randrange(4294967294))
    return int(seed)


def _prep_image(pil: Image.Image, height: int, width: int) -> tuple[Image.Image, int, int]:
    """Match generate_video.py: portrait images swap H/W, then resizecrop."""
    iw, ih = pil.size
    if ih > iw:
        height, width = width, height
    return resizecrop(pil, height, width), height, width


# ----------------------------- operations -----------------------------------

def op_image_to_video(inp: dict) -> dict:
    height, width = _dims(RESOLUTION)
    pil = _b64_to_pil(inp["image_base64"])
    pil, height, width = _prep_image(pil, height, width)
    seed = _resolve_seed(inp.get("seed"))
    num_frames = int(inp.get("num_frames", 97))
    fps = int(inp.get("fps", 24))
    pipe = get_pipeline("i2v", inp.get("model_id") or I2V_MODEL)

    kwargs = {
        "prompt": inp.get("motion_prompt") or inp.get("prompt") or "",
        "negative_prompt": inp.get("negative_prompt", DEFAULT_NEGATIVE),
        "num_frames": num_frames,
        "num_inference_steps": int(inp.get("inference_steps", 30)),
        "guidance_scale": float(inp.get("guidance_scale", 6.0)),
        "shift": float(inp.get("shift", 8.0)),
        "generator": torch.Generator(device="cuda").manual_seed(seed),
        "height": height,
        "width": width,
        "image": pil,
    }
    with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
        frames = pipe(**kwargs)[0]
    return {"video_base64": _frames_to_mp4_b64(frames, fps),
            "meta": {"operation": "image_to_video", "seed": seed,
                     "num_frames": num_frames, "fps": fps,
                     "resolution": RESOLUTION, "composite": bool(inp.get("composite", False))}}


def op_scene_continuation(inp: dict) -> dict:
    """Continue from the LAST FRAME of the previous clip (passed as image_base64).

    Uses the diffusion-forcing pipeline for stronger environmental/lighting
    continuity when available; otherwise falls back to image-to-video seeded
    with the last frame.
    """
    if not _HAS_DF:
        # Graceful fallback: last-frame -> i2v (still continuous, weaker coherence).
        out = op_image_to_video(inp)
        out["meta"]["operation"] = "scene_continuation"
        out["meta"]["mode"] = "i2v_fallback"
        return out

    height, width = _dims(RESOLUTION)
    pil = _b64_to_pil(inp["image_base64"])
    pil, height, width = _prep_image(pil, height, width)
    seed = _resolve_seed(inp.get("seed"))
    num_frames = int(inp.get("num_frames", 97))
    fps = int(inp.get("fps", 24))
    pipe = get_pipeline("df", inp.get("model_id") or DF_MODEL)

    kwargs = {
        "prompt": inp.get("motion_prompt") or inp.get("prompt") or "",
        "negative_prompt": inp.get("negative_prompt", DEFAULT_NEGATIVE),
        "image": pil,
        "num_frames": num_frames,
        "num_inference_steps": int(inp.get("inference_steps", 30)),
        "guidance_scale": float(inp.get("guidance_scale", 6.0)),
        "shift": float(inp.get("shift", 8.0)),
        "generator": torch.Generator(device="cuda").manual_seed(seed),
        "height": height,
        "width": width,
        "overlap_history": inp.get("overlap_history", 17),
        "addnoise_condition": int(inp.get("addnoise_condition", 20)),
        "base_num_frames": int(inp.get("base_num_frames", 97)),
        "ar_step": int(inp.get("ar_step", 0)),
        "causal_block_size": int(inp.get("causal_block_size", 1)),
    }
    with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
        frames = pipe(**kwargs)[0]
    return {"video_base64": _frames_to_mp4_b64(frames, fps),
            "meta": {"operation": "scene_continuation", "mode": "diffusion_forcing",
                     "seed": seed, "num_frames": num_frames, "fps": fps,
                     "resolution": RESOLUTION}}


def op_text_to_video(inp: dict) -> dict:
    height, width = _dims(RESOLUTION)
    seed = _resolve_seed(inp.get("seed"))
    num_frames = int(inp.get("num_frames", 97))
    fps = int(inp.get("fps", 24))
    pipe = get_pipeline("t2v", inp.get("model_id") or T2V_MODEL)

    kwargs = {
        "prompt": inp.get("prompt") or inp.get("motion_prompt") or "",
        "negative_prompt": inp.get("negative_prompt", DEFAULT_NEGATIVE),
        "num_frames": num_frames,
        "num_inference_steps": int(inp.get("inference_steps", 30)),
        "guidance_scale": float(inp.get("guidance_scale", 6.0)),
        "shift": float(inp.get("shift", 8.0)),
        "generator": torch.Generator(device="cuda").manual_seed(seed),
        "height": height,
        "width": width,
    }
    with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
        frames = pipe(**kwargs)[0]
    return {"video_base64": _frames_to_mp4_b64(frames, fps),
            "meta": {"operation": "text_to_video", "seed": seed,
                     "num_frames": num_frames, "fps": fps, "resolution": RESOLUTION}}


_OPS = {
    "image_to_video": op_image_to_video,
    "scene_continuation": op_scene_continuation,
    "text_to_video": op_text_to_video,
}


def handler(job: dict) -> dict:
    """RunPod serverless entrypoint. job = {"input": {...}}."""
    inp = job.get("input") or {}
    op = (inp.get("operation") or "").strip()
    if op not in _OPS:
        return {"error": f"unknown operation {op!r}. "
                         f"expected one of {sorted(_OPS)}"}
    if op in ("image_to_video", "scene_continuation") and not inp.get("image_base64"):
        return {"error": f"operation {op!r} requires 'image_base64'"}
    started = time.time()
    try:
        result = _OPS[op](inp)
        result.setdefault("meta", {})["duration_seconds"] = round(time.time() - started, 2)
        return result
    except Exception as exc:  # surface a clean error to the backend
        return {"error": str(exc), "trace": traceback.format_exc()[-1500:],
                "operation": op}


def _warm_start() -> None:
    """Pre-load weights into VRAM at container startup (per RunPod best practice)."""
    if PRELOAD in ("i2v", "t2v", "df"):
        model = {"i2v": I2V_MODEL, "t2v": T2V_MODEL, "df": DF_MODEL}[PRELOAD]
        print(f"[warm-start] loading {PRELOAD} pipeline: {model}", flush=True)
        try:
            get_pipeline(PRELOAD, model)
            print("[warm-start] pipeline resident in VRAM", flush=True)
        except Exception as exc:  # don't crash the worker; first call will retry
            print(f"[warm-start] deferred (will load on first call): {exc}", flush=True)


_warm_start()

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
