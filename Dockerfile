# SkyReels-V2 Video Engine — RunPod Serverless image
#
# Build-time fix: RunPod caps GitHub builds at 30 min. Compiling flash-attn from
# source (20-40 min) blew that limit. We install a PREBUILT flash-attn wheel
# instead (seconds), which only exists for cu12 + torch2.5 — so this image is
# built on CUDA 12.1 with torch 2.5.1+cu121 (was cu118).
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf/hub

# ubuntu 22.04 ships python 3.10 (matches the cp310 flash-attn wheel).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git curl ffmpeg build-essential && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) torch stack pinned to SkyReels' requirement, cu121 wheels.
RUN python -m pip install --upgrade pip && \
    python -m pip install torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121

# 2) PREBUILT flash-attn wheel (torch2.5 / cu12 / py310 / cxx11abiFALSE).
#    Install einops first, then the wheel with --no-deps so pip's resolver never
#    touches the already-installed cu121 torch (that resolver step is what failed
#    the RunPod build). Retry the 244MB download for transient network blips.
# pip parses version/tags from the wheel FILENAME, so save with the ORIGINAL
# name (with a literal '+'). Renaming to flash_attn.whl caused
# "Invalid wheel filename (wrong number of parts)"; -O would keep the URL's
# encoded '%2B'. So fetch the %2B URL but write the real '+' filename.
RUN python -m pip install einops && \
    curl -fSL --retry 4 --retry-delay 5 \
      -o "/tmp/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" && \
    python -m pip install --no-deps "/tmp/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" && \
    rm -f /tmp/flash_attn-2.8.3.post1*.whl

# 3) Repo requirements minus torch/torchvision/flash_attn (already installed).
COPY requirements.txt /app/requirements.txt
RUN grep -viE '^(torch==|torchvision==|flash_attn)' /app/requirements.txt > /app/req.trimmed.txt && \
    python -m pip install -r /app/req.trimmed.txt && \
    python -m pip install runpod pillow decord

# 4) App code (handler.py at repo root; skyreels_v2_infer importable).
COPY . /app

ENV RESOLUTION=540P \
    I2V_MODEL=Skywork/SkyReels-V2-I2V-1.3B-540P \
    T2V_MODEL=Skywork/SkyReels-V2-T2V-14B-540P \
    DF_MODEL=Skywork/SkyReels-V2-DF-1.3B-540P \
    PRELOAD=i2v \
    OFFLOAD=1 \
    ALLOW_MULTI_MODEL=0

CMD ["python", "-u", "handler.py"]
