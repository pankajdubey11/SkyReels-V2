# TrimTime Video Engine — SkyReels-V2 on RunPod Serverless (Phase 1)

Self-hosted image/text-to-video. **Zero external storage** — responses return the
`.mp4` as a Base64 string for the local backend + FFmpeg to assemble.

## Endpoints (dispatched by `input.operation`)

RunPod Serverless exposes ONE URL per endpoint; the local backend selects the
operation via the JSON body. Call `POST https://api.runpod.io/v2/<ENDPOINT_ID>/runsync`
(or `/run` for async) with `Authorization: Bearer <RUNPOD_API_KEY>`.

### 1. `image_to_video` — animate a keyframe / reference image
```json
{ "input": {
  "operation": "image_to_video",
  "image_base64": "<png/jpg base64>",
  "motion_prompt": "the barber lifts his scissors, warm golden light, subtle push-in",
  "num_frames": 81,            // ~5-10s at 24fps (spec default)
  "seed": 12345,               // integer -> locks the character across scenes
  "fps": 24,
  "resolution": "540P",        // optional; else server env RESOLUTION
  "composite": false           // set true when image is a multi-char canvas
}}
```

### 2. `scene_continuation` — continue from the LAST FRAME of the previous clip
```json
{ "input": {
  "operation": "scene_continuation",
  "image_base64": "<last frame of previous clip, base64>",
  "motion_prompt": "the camera pans right revealing the full salon",
  "num_frames": 81,
  "seed": 12345,
  "overlap_history": 17        // diffusion-forcing continuity window
}}
```
Uses the diffusion-forcing pipeline for environmental/lighting continuity when
available; otherwise falls back to image-to-video seeded with the last frame.

### 3. `text_to_video` — atmospheric / fallback (no reference image)
```json
{ "input": {
  "operation": "text_to_video",
  "prompt": "slow drifting clouds over a city at golden hour, cinematic",
  "num_frames": 97, "seed": 7, "fps": 24
}}
```

### Multi-character scenes (feature-blend prevention)
Build a **composite canvas** (Character A left, Character B right) into one image,
send it as `image_base64` on `image_to_video` with `composite: true`, and use
spatial language in the prompt:
> "the character on the LEFT in the blue coat talks to the character on the RIGHT in the red dress"

## Response
```json
{ "video_base64": "<mp4 base64>",
  "meta": { "operation": "...", "seed": 12345, "num_frames": 81,
            "fps": 24, "resolution": "540P", "duration_seconds": 42.1 } }
```
On failure: `{ "error": "...", "trace": "...", "operation": "..." }`.

## Deploy config (RunPod endpoint env vars)
| Env | Default | Notes |
|-----|---------|-------|
| `RESOLUTION` | `540P` | `540P` (544×960) or `720P` (720×1280) |
| `I2V_MODEL` | `Skywork/SkyReels-V2-I2V-1.3B-540P` | 1.3B fits smaller GPUs; use 14B for quality |
| `T2V_MODEL` | `Skywork/SkyReels-V2-T2V-14B-540P` | |
| `DF_MODEL` | `Skywork/SkyReels-V2-DF-1.3B-540P` | continuation |
| `PRELOAD` | `i2v` | pipeline warm-loaded at startup (`i2v`/`t2v`/`df`/`none`) |
| `OFFLOAD` | `1` | CPU-offload to save VRAM |
| `ALLOW_MULTI_MODEL` | `0` | keep >1 pipeline resident (needs big VRAM) |

**Weights:** mount a RunPod **network volume** at `/runpod-volume` (env `HF_HOME` points there) so weights persist across cold starts, OR bake into the image via `--build-arg BAKE_MODEL=...`.

**Warm start:** `handler.py` pre-loads the `PRELOAD` pipeline into VRAM at container startup (not per request), per RunPod best practice.

## Local smoke test (structure only; real inference needs a GPU)
```bash
# On a GPU box / inside the container:
python runpod/handler.py --test_input "$(cat runpod/test_input.json)"
```

## ⚠️ Build notes
- SkyReels pins `torch==2.5.1`; the Dockerfile reinstalls torch (cu118) over the
  2.1.0 base. If `flash-attn` fails to build, supply a prebuilt wheel matching
  **torch 2.5.1 / cu118 / py310** (this is the most common deployment snag).
- Cold start on 14B without a warm volume can be slow (large weight download).
  Prefer 1.3B for latency-sensitive scenes, or bake/volume the weights.
