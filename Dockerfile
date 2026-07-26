# SkyReels-V2 Video Engine — RunPod Serverless image
# Base per project spec. NOTE: SkyReels pins torch==2.5.1, so we reinstall torch
# (cu118 wheels) over the 2.1.0 base. flash-attn is built no-isolation to match.
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf/hub \
    PYTHONUNBUFFERED=1

# System deps: ffmpeg for imageio mp4 muxing, git for any VCS installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Pin torch stack to SkyReels' requirement (cu118).
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# 2) Repo requirements (minus torch/torchvision already installed). flash-attn
#    is heavy — installed separately with --no-build-isolation.
COPY requirements.txt /app/requirements.txt
RUN grep -viE '^(torch==|torchvision==|flash_attn)' /app/requirements.txt > /app/req.trimmed.txt && \
    pip install -r /app/req.trimmed.txt && \
    pip install runpod pillow && \
    pip install flash-attn --no-build-isolation || \
    echo "WARN: flash-attn build failed — set a prebuilt wheel matching torch2.5.1/cu118/py310"

# 3) App code (whole repo so skyreels_v2_infer is importable).
COPY . /app

# Optional: bake model weights into the image for faster cold starts.
#   docker build --build-arg BAKE_MODEL=Skywork/SkyReels-V2-I2V-1.3B-540P ...
# Default OFF — production should mount a RunPod network volume at /runpod-volume.
ARG BAKE_MODEL=""
RUN if [ -n "$BAKE_MODEL" ]; then \
      python -c "from skyreels_v2_infer.modules import download_model; download_model('$BAKE_MODEL')"; \
    fi

# Warm-load config (override per RunPod endpoint).
ENV RESOLUTION=540P \
    I2V_MODEL=Skywork/SkyReels-V2-I2V-1.3B-540P \
    T2V_MODEL=Skywork/SkyReels-V2-T2V-14B-540P \
    DF_MODEL=Skywork/SkyReels-V2-DF-1.3B-540P \
    PRELOAD=i2v \
    OFFLOAD=1 \
    ALLOW_MULTI_MODEL=0

CMD ["python", "-u", "handler.py"]
