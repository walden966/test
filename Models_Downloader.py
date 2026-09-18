"""
Model Downloader for Ultimate Video/Image Upscalers Premium

Downloads all required models:

SeedVR2 Core Models (from MonsterMMORPG/Wan_GGUF):
- VAE model (ema_vae_fp16.safetensors)
- SeedVR2 3B model (seedvr2_ema_3b_fp16.safetensors)
- SeedVR2 7B model (seedvr2_ema_7b_fp16.safetensors)
- SeedVR2 7B Sharp model (seedvr2_ema_7b_sharp_fp16.safetensors)

SeedVR2 FP8 Models (separate download group, from MonsterMMORPG/Wan_GGUF):
- SeedVR2 7B FP8 mixed_block35 model (seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors)
- SeedVR2 7B Sharp FP8 mixed_block35 model (seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors)

SeedVR2 GGUF Q8_0 Models (separate download group, from cmeka/SeedVR2-GGUF):
- SeedVR2 3B GGUF Q8_0 (seedvr2_ema_3b-Q8_0.gguf)
- SeedVR2 7B GGUF Q8_0 (seedvr2_ema_7b-Q8_0.gguf)
- SeedVR2 7B Sharp GGUF Q8_0 (seedvr2_ema_7b_sharp-Q8_0.gguf)

RIFE Models (from MonsterMMORPG/Wan_GGUF/RIFE_Models):
- RIFE folders: 4.14, 4.15, 4.17, 4.18, 4.20, 4.21, 4.22, 4.25, 4.26

BestImageUpscalers (from MonsterMMORPG/BestImageUpscalers):
- All model files for best image upscaling

FlashVSR+ Models:
- v1.0 from JunhaoZhuang/FlashVSR -> ComfyUI-FlashVSR_Stable/models/FlashVSR
- v1.1 from JunhaoZhuang/FlashVSR-v1.1 -> ComfyUI-FlashVSR_Stable/models/FlashVSR-v1.1
- posi_prompt.pth from MonsterMMORPG/Wan_GGUF -> ComfyUI-FlashVSR_Stable
- all files from MonsterMMORPG/Wan_GGUF/FlashVSR_VAEs -> both model folders above

SparkVSR Models:
- Stage-2 BF16 from MonsterMMORPG/Wan_GGUF/SparkVSR-bf16 -> SparkVSR/models/SparkVSR-bf16
- Stage-1 optional from JiongzeYu/SparkVSR-S1 -> SparkVSR/models/SparkVSR-S1
"""

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url, list_repo_files
from huggingface_hub import constants as hf_constants
from huggingface_hub.utils import build_hf_headers
import argparse
import collections
import concurrent.futures
import ctypes
import hashlib
import json
import logging
import os
import random
import re
import requests
import shutil
import sys
import threading
import time
import urllib.parse
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Keep downloader progress output stable on Windows consoles launched with
# legacy code pages, especially when progress bars contain Unicode glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    try:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# The hub and HTTP stacks log every request at INFO, which would bury the
# download progress this script prints.
for _noisy in ("huggingface_hub", "httpx", "httpcore", "urllib3", "requests", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Configuration - paths relative to script directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR_CANDIDATES = [
    "SECourses_Premium_Upscaler_Pro",
    "Ultimate_Video_Image_Upscalers_Premium",
]
PROJECT_DIR_NAME = next(
    (name for name in PROJECT_DIR_CANDIDATES if (SCRIPT_DIR / name).exists()),
    PROJECT_DIR_CANDIDATES[0],
)
PROJECT_DIR = SCRIPT_DIR / PROJECT_DIR_NAME
SEEDVR2_MODELS_DIR = PROJECT_DIR / "SeedVR2" / "models"
RIFE_MODELS_DIR = PROJECT_DIR / "RIFE" / "models"
IMAGE_UPSCALE_MODELS_DIR = PROJECT_DIR / "models"
FLASHVSR_MODELS_DIR = PROJECT_DIR / "ComfyUI-FlashVSR_Stable" / "models"
FLASHVSR_ROOT_DIR = PROJECT_DIR / "ComfyUI-FlashVSR_Stable"
FLASHVSR_INT8_CACHE_DIR = PROJECT_DIR / "FlashVSR_plus" / "models"
SPARKVSR_MODELS_DIR = PROJECT_DIR / "SparkVSR" / "models"
LTX25_MODELS_DIR = PROJECT_DIR / "LTX25_Models"
CACHE_DIR = SCRIPT_DIR / "download_cache"

# Repository configuration
SEEDVR2_REPO_ID = "MonsterMMORPG/Wan_GGUF"
SEEDVR2_GGUF_REPO_ID = "cmeka/SeedVR2-GGUF"
RIFE_REPO_ID = "MonsterMMORPG/Wan_GGUF"
RIFE_REPO_SUBDIR = "RIFE_Models"
RIFE_VERSION_FOLDERS = ["4.14", "4.15", "4.17", "4.18", "4.20", "4.21", "4.22", "4.25", "4.26"]
RIFE_REPO_PREFIXES = [f"{RIFE_REPO_SUBDIR}/{version}" for version in RIFE_VERSION_FOLDERS]
BESTIMAGEUPSCALE_REPO_ID = "MonsterMMORPG/BestImageUpscalers"
FLASHVSR_V10_REPO_ID = "JunhaoZhuang/FlashVSR"
FLASHVSR_V11_REPO_ID = "JunhaoZhuang/FlashVSR-v1.1"
FLASHVSR_VAE_REPO_ID = "MonsterMMORPG/Wan_GGUF"
FLASHVSR_VAE_SUBDIR = "FlashVSR_VAEs"
SPARKVSR_S2_REPO_ID = "MonsterMMORPG/Wan_GGUF"
SPARKVSR_S2_REPO_SUBDIR = "SparkVSR-bf16"
SPARKVSR_S1_REPO_ID = "JiongzeYu/SparkVSR-S1"
LTX25_REPO_ID = "MonsterMMORPG/Wan_GGUF"
LTX25_INT4_REPO_ID = "tsolful/LTX_2.5_INT4_W4A8_ConvRot"
LTX25_IC_LORA_REPO_ID = "MonsterMMORPG/Wan_GGUF"

PREBUILT_INT8_CONVROT_FILES = {
    "FlashVSR": "FlashVSR_int8_convrot.safetensors",
    "FlashVSR-v1.1": "FlashVSR-v1.1_int8_convrot.safetensors",
    "SparkVSR": "SparkVSR-int8-convrot.safetensors",
    "seedvr2_ema_3b_fp16.safetensors": "seedvr2_ema_3b_fp16_int8_convrot.safetensors",
    "seedvr2_ema_7b_fp16.safetensors": "seedvr2_ema_7b_fp16_int8_convrot.safetensors",
    "seedvr2_ema_7b_sharp_fp16.safetensors": "seedvr2_ema_7b_sharp_fp16_int8_convrot.safetensors",
}

FLASHVSR_VAE_FILES = {
    "Wan2.1": (None, "Wan2.1_VAE.pth"),
    "Wan2.2": (FLASHVSR_VAE_REPO_ID, "FlashVSR_VAEs/Wan2.2_VAE.pth"),
    "LightVAE_W2.1": (FLASHVSR_VAE_REPO_ID, "FlashVSR_VAEs/lightvaew2_1.pth"),
    "TAE_W2.2": (FLASHVSR_VAE_REPO_ID, "FlashVSR_VAEs/taew2_2.safetensors"),
    "LightTAE_HY1.5": (FLASHVSR_VAE_REPO_ID, "FlashVSR_VAEs/lighttaehy1_5.pth"),
}

BESTIMAGEUPSCALE_MODEL_FILES = (
    "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors",
    "2xLiveActionV1_SPAN_490000.pth",
    "2xNomosUni_span_multijpg_ldl.pth",
    "2x_AniScale2_Omni_i16_40K.pth",
    "4x-AnimeSharp.safetensors",
    "4x-UltraSharp.safetensors",
    "4x-UltraSharpV2.safetensors",
    "4xNomos2_hq_dat2.safetensors",
    "4xRealWebPhoto_v4_dat2.safetensors",
    "HAT-L_SRx4_ImageNet-pretrain.safetensors",
    "Kim2091-4x-UltraSharp.safetensors",
    "RealESRGAN_x4plus.safetensors",
    "RealESRGAN_x4plus_anime_6B.safetensors",
)

# SeedVR2 Model configurations
SEEDVR2_MODEL_CONFIGS = {
    "vae": {
        "filename": "ema_vae_fp16.safetensors",
        "name": "VAE Model (FP16)",
        "description": "Variational Autoencoder for SeedVR2",
        "size_hint": "~330 MB",
    },
    "3b": {
        "filename": "seedvr2_ema_3b_fp16.safetensors",
        "name": "SeedVR2 3B Model (FP16)",
        "description": "3 billion parameter model - faster, lower VRAM",
        "size_hint": "~6.5 GB",
    },
    "7b": {
        "filename": "seedvr2_ema_7b_fp16.safetensors",
        "name": "SeedVR2 7B Model (FP16)",
        "description": "7 billion parameter model - best quality",
        "size_hint": "~14 GB",
    },
    "7b_sharp": {
        "filename": "seedvr2_ema_7b_sharp_fp16.safetensors",
        "name": "SeedVR2 7B Sharp Model (FP16)",
        "description": "7 billion parameter model - sharpened variant",
        "size_hint": "~14 GB",
    },
    "7b_fp8_mixed_block35": {
        "filename": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        "name": "SeedVR2 7B FP8 Mixed Block35 Model",
        "description": "7 billion parameter model - FP8 e4m3fn mixed block35 variant",
        "size_hint": "~14 GB",
    },
    "7b_sharp_fp8_mixed_block35": {
        "filename": "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        "name": "SeedVR2 7B Sharp FP8 Mixed Block35 Model",
        "description": "7 billion parameter sharpened model - FP8 e4m3fn mixed block35 variant",
        "size_hint": "~14 GB",
    },
    "3b_gguf_q8_0": {
        "filename": "seedvr2_ema_3b-Q8_0.gguf",
        "name": "SeedVR2 3B GGUF Q8_0 Model",
        "description": "3 billion parameter GGUF Q8_0 quantized model",
        "size_hint": "~4 GB",
        "repo_id": SEEDVR2_GGUF_REPO_ID,
    },
    "7b_gguf_q8_0": {
        "filename": "seedvr2_ema_7b-Q8_0.gguf",
        "name": "SeedVR2 7B GGUF Q8_0 Model",
        "description": "7 billion parameter GGUF Q8_0 quantized model",
        "size_hint": "~8 GB",
        "repo_id": SEEDVR2_GGUF_REPO_ID,
    },
    "7b_sharp_gguf_q8_0": {
        "filename": "seedvr2_ema_7b_sharp-Q8_0.gguf",
        "name": "SeedVR2 7B Sharp GGUF Q8_0 Model",
        "description": "7 billion parameter sharpened GGUF Q8_0 quantized model",
        "size_hint": "~8 GB",
        "repo_id": SEEDVR2_GGUF_REPO_ID,
    },
}

CORE_SEEDVR2_MODEL_IDS = [
    "vae",
    "3b",
    "7b",
    "7b_sharp",
]

FP8_SEEDVR2_MODEL_IDS = [
    "7b_fp8_mixed_block35",
    "7b_sharp_fp8_mixed_block35",
]

GGUF_SEEDVR2_MODEL_IDS = [
    "3b_gguf_q8_0",
    "7b_gguf_q8_0",
    "7b_sharp_gguf_q8_0",
]

# "Regular all" should exclude FP8 and GGUF.
ALL_SEEDVR2_MODEL_IDS = CORE_SEEDVR2_MODEL_IDS.copy()

# Dedicated separate downloader group.
SEPARATE_GGUF_FP8_MODEL_IDS = FP8_SEEDVR2_MODEL_IDS + GGUF_SEEDVR2_MODEL_IDS

# RIFE configuration - downloads selected folders from Wan_GGUF/RIFE_Models
RIFE_CONFIG = {
    "name": "RIFE Models (4.14-4.26)",
    "description": "Frame interpolation models for RIFE",
    "repo_id": RIFE_REPO_ID,
    "repo_subdir": RIFE_REPO_SUBDIR,
    "versions": RIFE_VERSION_FOLDERS,
    "target_dir": RIFE_MODELS_DIR,
}

BESTIMAGEUPSCALE_CONFIG = {
    "name": "BestImageUpscalers Models",
    "description": "High-quality image upscaling models",
    "repo_id": BESTIMAGEUPSCALE_REPO_ID,
    "target_dir": IMAGE_UPSCALE_MODELS_DIR,
}

FLASHVSR_CONFIGS = {
    "1.0": {
        "name": "FlashVSR+ v1.0",
        "description": "FlashVSR+ model weights for FlashVSR runtime folder",
        "repo_id": FLASHVSR_V10_REPO_ID,
        "target_dir": FLASHVSR_MODELS_DIR / "FlashVSR",
    },
    "1.1": {
        "name": "FlashVSR+ v1.1",
        "description": "FlashVSR+ model weights for FlashVSR-v1.1 runtime folder",
        "repo_id": FLASHVSR_V11_REPO_ID,
        "target_dir": FLASHVSR_MODELS_DIR / "FlashVSR-v1.1",
    },
}

FLASHVSR_AUXILIARY_FILES = [
    {
        "name": "FlashVSR+ posi_prompt",
        "description": "Additional FlashVSR+ prompt model",
        "repo_id": SEEDVR2_REPO_ID,
        "filename": "posi_prompt.pth",
        "target_dir": FLASHVSR_ROOT_DIR,
    },
]

FLASHVSR_VAE_PREFIXES = [FLASHVSR_VAE_SUBDIR]

SPARKVSR_CONFIGS = {
    "s2": {
        "name": "SparkVSR Stage-2 BF16",
        "description": "SparkVSR BF16 Diffusers model folder from MonsterMMORPG/Wan_GGUF",
        "repo_id": SPARKVSR_S2_REPO_ID,
        "repo_subdir": SPARKVSR_S2_REPO_SUBDIR,
        "include_prefixes": [SPARKVSR_S2_REPO_SUBDIR],
        "strip_prefix": SPARKVSR_S2_REPO_SUBDIR,
        "target_dir": SPARKVSR_MODELS_DIR / SPARKVSR_S2_REPO_SUBDIR,
    },
    "s1": {
        "name": "SparkVSR Stage-1",
        "description": "Official SparkVSR Stage-1 Diffusers model",
        "repo_id": SPARKVSR_S1_REPO_ID,
        "target_dir": SPARKVSR_MODELS_DIR / "SparkVSR-S1",
    },
}

# ---------------------------------------------------------------------------
# LTX 2.5 Upscaler — every file lands flat inside LTX25_Models so users can
# find, inspect and reuse the checkpoints directly.
# Keep the name maps in sync with SECourses_Premium_Upscaler_Pro/shared/ltx25_constants.py.
# ---------------------------------------------------------------------------
LTX25_FILE_SOURCES = {
    "LTX-2.5-22b-Distilled-Transformer-Int8-ConvRot-Premium-SECourses.safetensors": LTX25_REPO_ID,
    "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors": LTX25_REPO_ID,
    "ltx-2.5-22b-distilled-transformer_W4A8_Mixed.safetensors": LTX25_INT4_REPO_ID,
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors": LTX25_REPO_ID,
    "LTX-2.5-22b-Dev-Transformer-Int8-ConvRot-Premium-SECourses.safetensors": LTX25_REPO_ID,
    "ltx-2.5-22b-dev-transformer_W4A8_Mixed.safetensors": LTX25_INT4_REPO_ID,
    "ltx-2.5-22b-dev-transformer-bf16.safetensors": LTX25_REPO_ID,
    "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors": LTX25_REPO_ID,
    "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors": LTX25_REPO_ID,
    "ltx-2.5-video-vae-conv-bf16.safetensors": LTX25_REPO_ID,
    "ltx-2.5-video-vae-bf16.safetensors": LTX25_REPO_ID,
    "ltx-2.5-audio-vae-bf16.safetensors": LTX25_REPO_ID,
    "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors": LTX25_IC_LORA_REPO_ID,
    "ltx-2.5-22b-distilled-lora-450-bf16.safetensors": LTX25_REPO_ID,
    "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors": LTX25_REPO_ID,
    "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors": LTX25_REPO_ID,
    "ltx-2.5-duration-head-bf16.safetensors": LTX25_REPO_ID,
}

LTX25_TRANSFORMERS = {
    "Distilled INT8 ConvRot": "LTX-2.5-22b-Distilled-Transformer-Int8-ConvRot-Premium-SECourses.safetensors",
    "Distilled NVFP4": "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
    "Distilled INT4 W4A8 ConvRot": "ltx-2.5-22b-distilled-transformer_W4A8_Mixed.safetensors",
    "Distilled BF16": "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "Dev INT8 ConvRot": "LTX-2.5-22b-Dev-Transformer-Int8-ConvRot-Premium-SECourses.safetensors",
    "Dev INT4 W4A8 ConvRot": "ltx-2.5-22b-dev-transformer_W4A8_Mixed.safetensors",
    "Dev BF16": "ltx-2.5-22b-dev-transformer-bf16.safetensors",
}

LTX25_TEXT_ENCODERS = {
    "Gemma 4 12B INT8 ConvRot": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "Gemma 4 12B BF16": "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
}

LTX25_VIDEO_VAES = {
    "Video VAE Conv": "ltx-2.5-video-vae-conv-bf16.safetensors",
    "Video VAE Regular": "ltx-2.5-video-vae-bf16.safetensors",
}

# Needed by every LTX 2.5 upscale run regardless of the selected variant.
LTX25_ALWAYS_FILES = (
    "ltx-2.5-audio-vae-bf16.safetensors",
    "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
)

# Optional extras for power users; downloaded only by the bulk --ltx25 flag.
LTX25_EXTRA_FILES = (
    "ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
    "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
    "ltx-2.5-duration-head-bf16.safetensors",
)

# =============================================================================
# Transfer engine
# =============================================================================

DOWNLOAD_CONFIG = {
    # HTTP starts conservatively and scales only for a genuinely large queue.
    # Native Xet remains the preferred path for Xet-backed weights.  The HTTP
    # fallback therefore optimizes for portable resumes and non-Xet servers,
    # where opening 96 sockets was slower on both the Linux benchmark host and
    # typical consumer Windows machines.
    "num_connections": 16,
    "http_adaptive_connections": 1,
    "http_max_connections": 32,
    # A file is cut into pieces and every piece is fetched with its own range
    # request. Giving each connection one enormous range instead lets a single
    # shaped or unlucky flow decide the speed of the whole file. Pieces scale
    # with the file, from this size up to the cap while aiming for the target
    # count, so a 15 GB model spends its time moving bytes instead of opening
    # requests while a 100 MB file keeps fine-grained pieces and retries.
    "piece_size": 16 * 1024 * 1024,
    "piece_size_max": 32 * 1024 * 1024,
    "pieces_per_file_target": 512,
    # The model CDN is fast when a flow is fresh but can shape a long-lived
    # flow heavily. Rotate after a couple of ranges and stagger the time limit
    # so the fleet never falls into one synchronized slow wave.
    "pieces_per_connection": 2,
    "max_connection_seconds": 30.0,
    "connection_age_jitter": 0.35,
    # Prefer Hugging Face's native Rust/Xet range stream for large Xet-backed
    # files. It remains optional: the HTTP range engine is the complete fallback
    # on older installs and for highly fragmented resumes.
    "xet_enabled": 1,
    "xet_threshold": 256 * 1024 * 1024,
    "xet_concurrency": 32,
    # Zero means unlimited.  A single Xet stream group is reused for every
    # missing interval, so a fragmented resume stays on Xet instead of falling
    # back to HTTP merely because an interruption left many small gaps.
    "xet_max_resume_gaps": 0,
    "xet_range_flush_size": 32 * 1024 * 1024,
    "xet_high_memory_threshold": 64 * 1024 * 1024 * 1024,
    # After this fair chance, a connection moving less than this fraction of
    # the current per-connection fleet average is dropped and dialed again.
    # The absolute floor further down catches the whole fleet crawling.
    "slow_connection_grace": 8.0,
    "slow_connection_fraction": 0.35,
    "read_block_size": 1024 * 1024,
    # Received bytes are batched to this size before they reach the file, which
    # keeps writes large and, because pieces are handed out in order, close
    # together on disk.
    "write_block_size": 4 * 1024 * 1024,
    # Files below this size use one connection, but several files run at once.
    "parallel_threshold": 16 * 1024 * 1024,
    "small_file_workers": 8,
    # Hashing below this size finishes before a progress line would help.
    "verify_progress_threshold": 64 * 1024 * 1024,
    "metadata_workers": 16,
    "max_retries": 8,
    "retry_delay": 1.0,
    "max_retry_delay": 20.0,
    "connect_timeout": 15.0,
    "read_timeout": 30.0,
    "metadata_timeout": 30.0,
    # Signed CDN links expire; they are refreshed well before that happens.
    "url_refresh_seconds": 900.0,
    "state_flush_interval": 2.0,
    # Speed is reported over this sliding window instead of since the start, so
    # the number on screen is the speed right now.
    "speed_window": 12.0,
    "progress_interval": 0.4,
    "non_tty_progress_interval": 5.0,
    "hash_block_size": 8 * 1024 * 1024,
    # A connection still below this rate after that many seconds is dropped and
    # replaced rather than allowed to hold the file back.
    "min_piece_rate": 128 * 1024,
    "min_piece_seconds": 20.0,
    # Whole-file guards: restart the transfer this many times, and give up only
    # after nothing at all has arrived for this long.
    "file_attempts": 6,
    "stall_timeout": 300.0,
}


# Any setting above can be overridden for a single run, which is how values are
# compared on the machine that actually shows the problem rather than guessed:
#   SECOURSES_DL_MAX_CONNECTION_SECONDS=3 python Models_Downloader.py --7b-sharp
for _key, _default in list(DOWNLOAD_CONFIG.items()):
    _override = os.environ.get("SECOURSES_DL_" + _key.upper())
    if _override:
        with suppress(ValueError):
            DOWNLOAD_CONFIG[_key] = type(_default)(_override)

RESUME_STATE_VERSION = 2

_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)
_HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_PART_RE = re.compile(r"\.part(\d+)$")

_HF_ENDPOINT = str(getattr(hf_constants, "ENDPOINT", "https://huggingface.co"))
_HF_ENDPOINT_HOST = (urllib.parse.urlsplit(_HF_ENDPOINT).hostname or "huggingface.co").lower()


class DownloadError(RuntimeError):
    """A transfer failed in a way the caller should report."""


class RangeNotSupported(DownloadError):
    """The server refused to serve byte ranges."""


class _TooSlow(DownloadError):
    """A connection fell far behind the fleet and is redialed immediately."""


class _Cancelled(RuntimeError):
    """Raised inside workers once the file has been abandoned."""


@dataclass(frozen=True)
class RemoteFile:
    """Everything needed to fetch and verify one repository file."""

    repo_id: str
    filename: str
    size: int
    digest_algorithm: Optional[str]
    digest: Optional[str]
    commit_hash: str

    @property
    def key(self) -> Tuple[str, str]:
        return (self.repo_id, self.filename)


def _digest_from_etag(etag: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Turn a Hugging Face etag or blob id into a verifiable digest.

    LFS weights expose a plain sha256. Small files such as config.json are
    stored in git, so their id is the sha1 of ``blob <size>\\0<content>``.
    """
    if not etag:
        return None, None
    value = etag.replace('"', "").replace("W/", "").strip().lower()
    if value.startswith("sha256:"):
        value = value[7:]
    if _HEX_64_RE.match(value):
        return "sha256", value
    if _HEX_40_RE.match(value):
        return "git-sha1", value
    return None, None


def _reserve_blocks(path: Path, size: int) -> bool:
    """Claim the whole file from the allocator before anything is written.

    A staging file that grows while a dozen connections write into it makes the
    file system extend its block map over and over, and every later write pays
    for the longer map. Reserving the space once keeps that cost flat for the
    whole transfer. ``fallocate`` is called directly rather than through
    ``os.posix_fallocate`` because the libc wrapper quietly falls back to
    writing gigabytes of zeroes on file systems that cannot do this.
    """
    if not sys.platform.startswith("linux") or size <= 0:
        return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.fallocate.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong
        ]
        fd = os.open(str(path), os.O_WRONLY)
        try:
            return libc.fallocate(fd, 0, 0, size) == 0
        finally:
            os.close(fd)
    except (OSError, AttributeError, ValueError):
        return False


def _enable_sparse_file(path: Path) -> bool:
    """Keep a staging file sparse so out-of-order writes stay cheap.

    Ranges land all over the file, and on NTFS the first write near the end of a
    15 GB file otherwise forces Windows to zero-fill everything before it.
    POSIX file systems already behave this way.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes.wintypes as wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.DeviceIoControl.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        kernel32.DeviceIoControl.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateFileW(
            str(path),
            0x40000000,  # GENERIC_WRITE
            0x00000003,  # FILE_SHARE_READ | FILE_SHARE_WRITE
            None,
            3,  # OPEN_EXISTING
            0x80,  # FILE_ATTRIBUTE_NORMAL
            None,
        )
        if handle == wintypes.HANDLE(-1).value:
            return False
        try:
            returned = wintypes.DWORD(0)
            return bool(
                kernel32.DeviceIoControl(
                    handle,
                    0x000900C4,  # FSCTL_SET_SPARSE
                    None,
                    0,
                    None,
                    0,
                    ctypes.byref(returned),
                    None,
                )
            )
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, ValueError, AttributeError, ImportError):
        return False


def _allocated_file_bytes(path: Path) -> int:
    """Return physical bytes occupied by ``path`` as portably as possible.

    ``st_size`` is the logical length and therefore reports a newly-created
    50 GB sparse staging file as full even when almost no blocks are allocated.
    POSIX exposes 512-byte block counts; Windows exposes the equivalent through
    GetCompressedFileSizeW.  Falling back to the logical size is conservative
    for non-sparse/unknown file systems.
    """
    try:
        stat = path.stat()
    except OSError:
        return 0

    if os.name == "nt":
        try:
            import ctypes.wintypes as wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCompressedFileSizeW.argtypes = [
                wintypes.LPCWSTR,
                ctypes.POINTER(wintypes.DWORD),
            ]
            kernel32.GetCompressedFileSizeW.restype = wintypes.DWORD
            high = wintypes.DWORD(0)
            ctypes.set_last_error(0)
            low = int(kernel32.GetCompressedFileSizeW(str(path), ctypes.byref(high)))
            if low == 0xFFFFFFFF and ctypes.get_last_error() != 0:
                raise OSError(ctypes.get_last_error(), "GetCompressedFileSizeW failed")
            return min(stat.st_size, (int(high.value) << 32) | low)
        except (OSError, ValueError, AttributeError, ImportError):
            return stat.st_size

    blocks = getattr(stat, "st_blocks", None)
    if isinstance(blocks, int) and blocks >= 0:
        return min(stat.st_size, blocks * 512)
    return stat.st_size


def _total_memory_bytes() -> int:
    """Best-effort physical-memory size without making psutil mandatory."""
    with suppress(Exception):
        import psutil

        return int(psutil.virtual_memory().total)
    if os.name == "posix":
        with suppress(OSError, ValueError, AttributeError):
            return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    if os.name == "nt":
        with suppress(Exception):
            import ctypes.wintypes as wintypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", wintypes.DWORD),
                    ("memory_load", wintypes.DWORD),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical)
    return 0


def _fsync_parent(path: Path) -> None:
    """Durably record a rename on file systems which support directory fsync."""
    if os.name != "posix":
        return
    descriptor = None
    try:
        descriptor = os.open(str(path.parent), os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


class _SpeedMeter:
    """Throughput over a sliding window rather than since the download began."""

    def __init__(self, window: float):
        self._window = max(1.0, float(window))
        self._lock = threading.Lock()
        self._total = 0
        self._samples = collections.deque()

    def add(self, count: int) -> None:
        with self._lock:
            self._total += count

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def rate(self) -> float:
        now = time.monotonic()
        with self._lock:
            total = self._total
            self._samples.append((now, total))
            while len(self._samples) > 2 and now - self._samples[0][0] > self._window:
                self._samples.popleft()
            oldest_at, oldest_total = self._samples[0]
        span = now - oldest_at
        if span <= 0.05:
            return 0.0
        return max(0.0, (total - oldest_total) / span)


class _Link:
    """One worker's HTTP session, replaced when it misbehaves or grows old.

    A connection that has just been slow or failed rarely improves, so it is
    thrown away and dialed again. Healthy sockets are reused briefly to avoid
    needless handshakes, then rotated before CDN shaping can drag down a whole
    synchronized fleet.
    """

    __slots__ = (
        "_factory",
        "_limit",
        "_max_age",
        "_age_jitter",
        "_session",
        "_used",
        "_expires",
    )

    def __init__(self, factory, limit: int, max_age: float, age_jitter: float = 0.0):
        self._factory = factory
        self._limit = max(1, int(limit))
        self._max_age = max(1.0, float(max_age))
        self._age_jitter = min(0.9, max(0.0, float(age_jitter)))
        self._session = None
        self._used = 0
        self._expires = 0.0

    def session(self):
        # Both limits here are backstops. A genuinely slow connection is
        # noticed against the rest of the fleet inside _fetch_piece and
        # dropped long before either arm triggers.
        now = time.monotonic()
        if (
            self._session is None
            or self._used >= self._limit
            or now >= self._expires
        ):
            self.drop()
            self._session = self._factory()
            self._used = 0
            spread = self._max_age * self._age_jitter
            self._expires = now + random.uniform(
                self._max_age - spread,
                self._max_age + spread,
            )
        self._used += 1
        return self._session

    def drop(self) -> None:
        if self._session is not None:
            with suppress(Exception):
                self._session.close()
            self._session = None
            self._expires = 0.0


class _Progress:
    """When bytes last reached the file, and how many connections are moving."""

    __slots__ = ("_lock", "last", "active")

    def __init__(self):
        self._lock = threading.Lock()
        self.last = time.monotonic()
        self.active = 0

    def touch(self) -> None:
        self.last = time.monotonic()

    def enter(self) -> None:
        with self._lock:
            self.active += 1

    def leave(self) -> None:
        with self._lock:
            self.active -= 1


class _Ranges:
    """The byte ranges of a staging file that are already on disk.

    This is the whole resume story: half-open ``[start, end)`` intervals, merged
    as they arrive and saved next to the staging file. It does not care which
    connection wrote what or how the file was cut up, so a download started by
    one version resumes under another, and an interrupted run continues from the
    exact byte it reached rather than from a piece or segment boundary.
    """

    __slots__ = ("_items", "_lock", "_total")

    def __init__(self, items: Optional[Iterable[Sequence[int]]] = None):
        self._lock = threading.Lock()
        self._items: List[List[int]] = []
        self._total = 0
        for start, end in items or []:
            self.add(int(start), int(end))

    def add(self, start: int, end: int) -> None:
        self.add_many(((start, end),))

    def add_many(self, items: Iterable[Sequence[int]]) -> None:
        """Merge several completed ranges in one pass.

        Xet can return thousands of sub-megabyte chunks out of order. Batching
        their interval updates avoids sorting the complete map once per chunk.
        """
        additions = [
            [int(start), int(end)]
            for start, end in items
            if int(end) > int(start)
        ]
        if not additions:
            return
        with self._lock:
            merged: List[List[int]] = []
            for item in sorted(self._items + additions):
                if merged and item[0] <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], item[1])
                else:
                    merged.append(item)
            self._items = merged
            self._total = sum(stop - begin for begin, stop in merged)

    def prefix(self, start: int, limit: int) -> int:
        """How many bytes from ``start`` are already present, up to ``limit``."""
        with self._lock:
            for begin, stop in self._items:
                if begin > start:
                    break
                if start < stop:
                    return min(stop - start, limit)
        return 0

    def missing(self, start: int, stop: int) -> List[Tuple[int, int]]:
        """Return the exact gaps in ``[start, stop)`` that are not on disk."""
        if stop <= start:
            return []
        with self._lock:
            cursor = start
            gaps: List[Tuple[int, int]] = []
            for begin, end in self._items:
                if end <= cursor:
                    continue
                if begin >= stop:
                    break
                if begin > cursor:
                    gaps.append((cursor, min(begin, stop)))
                cursor = max(cursor, min(end, stop))
                if cursor >= stop:
                    break
            if cursor < stop:
                gaps.append((cursor, stop))
            return gaps

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def snapshot(self) -> List[List[int]]:
        with self._lock:
            return [list(item) for item in self._items]


_REPO_INDEX_CACHE: Dict[str, Tuple[str, Dict[str, RemoteFile]]] = {}
_REPO_INDEX_LOCK = threading.Lock()


def repo_index(repo_id: str) -> Tuple[str, Dict[str, RemoteFile]]:
    """List a repository once, with the size and digest of every file.

    One ``repo_info`` call replaces a per-file metadata request, which for a
    repository with hundreds of entries is the difference between one second and
    several minutes before the first byte is transferred.
    """
    with _REPO_INDEX_LOCK:
        cached = _REPO_INDEX_CACHE.get(repo_id)
    if cached is not None:
        return cached

    info = HfApi().repo_info(repo_id, files_metadata=True)
    commit_hash = str(getattr(info, "sha", "") or "main")
    files: Dict[str, RemoteFile] = {}
    for sibling in getattr(info, "siblings", None) or []:
        name = getattr(sibling, "rfilename", None)
        size = getattr(sibling, "size", None)
        if not name or not isinstance(size, int):
            continue
        lfs = getattr(sibling, "lfs", None)
        etag = getattr(lfs, "sha256", None) if lfs is not None else None
        if not etag:
            etag = getattr(sibling, "blob_id", None)
        algorithm, digest = _digest_from_etag(etag)
        files[name] = RemoteFile(repo_id, name, size, algorithm, digest, commit_hash)

    result = (commit_hash, files)
    with _REPO_INDEX_LOCK:
        _REPO_INDEX_CACHE[repo_id] = result
    return result


def list_repo_file_names(repo_id: str) -> List[str]:
    """Repository file list, reusing the cached index when one exists."""
    try:
        return sorted(repo_index(repo_id)[1])
    except Exception:
        return sorted(list_repo_files(repo_id))


class RobustDownloader:
    """Resumable, verified downloader built on parallel byte ranges.

    A large file is cut into pieces that many connections claim in file
    order, each piece fetched with its own range request. Nothing
    depends on a single connection keeping its speed, so one shaped or
    unlucky flow costs a piece rather than the rest of the download, and the
    connections keep writing to one moving point in the file instead of a dozen
    places gigabytes apart.

    Bytes land in a sparse ``.part`` file at absolute offsets and the file is
    installed only once its digest matches the repository. Every byte that
    reaches disk is recorded in a map saved alongside it, so an interrupted run
    resumes at exactly the byte it reached.
    """

    def __init__(self, config: Dict, skip_verify: bool = False):
        self.config = dict(DOWNLOAD_CONFIG)
        self.config.update(config or {})
        self.skip_verify = skip_verify

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.verified_cache_file = CACHE_DIR / "verified_files_cache.json"
        self.verified_cache = self._load_json_cache(self.verified_cache_file)
        self._cache_lock = threading.Lock()

        self._progress_lock = threading.Lock()
        self._active_progress = False
        self._last_progress_len = 0
        try:
            self._interactive = bool(sys.stdout.isatty())
        except (AttributeError, OSError):
            self._interactive = False
        self._last_throttled_at = 0.0

        self._remote_cache: Dict[Tuple[str, str], RemoteFile] = {}
        self._xet_cache: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self._remote_lock = threading.Lock()
        self._url_cache: Dict[Tuple[str, str], Tuple[str, float]] = {}
        self._url_refreshing: set = set()
        self._url_lock = threading.Lock()

        self._hf_headers = dict(build_hf_headers())

    # ------------------------------- plumbing --------------------------------

    def _create_session(self, pool: int = 4) -> requests.Session:
        """A session with its own connection pool.

        Transfer workers each hold one and replace it periodically, so their
        sockets are never reused for longer than a handful of pieces. Retries
        are handled per piece, so the adapter itself does not retry.
        """
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool,
            pool_maxsize=pool,
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _load_json_cache(self, filepath: Path) -> Dict:
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return data if isinstance(data, dict) else {}
            except (IOError, OSError, json.JSONDecodeError) as exc:
                logger.warning(f"Could not load cache {filepath.name}: {exc}")
        return {}

    def _save_json_cache(self, filepath: Path, data: Dict) -> None:
        try:
            temp_file = filepath.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            temp_file.replace(filepath)
        except (IOError, OSError) as exc:
            logger.warning(f"Could not save cache {filepath.name}: {exc}")

    def _get_terminal_width(self) -> int:
        try:
            return shutil.get_terminal_size(fallback=(100, 20)).columns
        except Exception:
            return 100

    def show_progress_line(self, text: str) -> None:
        with self._progress_lock:
            if not self._interactive:
                # Piped output (the app runs this downloader as a subprocess)
                # gets whole lines at a slow cadence instead of carriage returns.
                now = time.monotonic()
                interval = max(0.1, float(self.config["non_tty_progress_interval"]))
                if now - self._last_throttled_at >= interval:
                    print(text, flush=True)
                    self._last_throttled_at = now
                return
            text = text[: max(1, self._get_terminal_width() - 1)]
            sys.stdout.write("\r" + text)
            if self._last_progress_len > len(text):
                sys.stdout.write(" " * (self._last_progress_len - len(text)))
                sys.stdout.write("\r" + text)
            sys.stdout.flush()
            self._last_progress_len = len(text)
            self._active_progress = True

    def finalize_progress_line(self, final_text: Optional[str] = None) -> None:
        with self._progress_lock:
            if not self._interactive:
                if final_text is not None:
                    print(final_text, flush=True)
                self._last_throttled_at = 0.0
                self._last_progress_len = 0
                self._active_progress = False
                return
            if final_text is not None:
                final_text = final_text[: max(1, self._get_terminal_width() - 1)]
                sys.stdout.write("\r" + final_text)
                if self._last_progress_len > len(final_text):
                    sys.stdout.write(" " * (self._last_progress_len - len(final_text)))
                sys.stdout.write("\n")
                sys.stdout.flush()
            elif self._active_progress:
                sys.stdout.write("\n")
                sys.stdout.flush()
            self._last_progress_len = 0
            self._active_progress = False

    def log(self, msg: str) -> None:
        with self._progress_lock:
            if self._active_progress:
                sys.stdout.write("\r" + " " * max(self._last_progress_len, 1) + "\r")
                self._last_progress_len = 0
                self._active_progress = False
            print(msg, flush=True)

    @staticmethod
    def format_bytes(bytes_val: float) -> str:
        bytes_val = max(0.0, float(bytes_val))
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} TB"

    @staticmethod
    def format_time(seconds: float) -> str:
        if seconds < 0 or seconds != seconds:
            return "0s"
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            mins, secs = divmod(seconds, 60)
            return f"{mins}m {secs}s" if secs else f"{mins}m"
        hours, rest = divmod(seconds, 3600)
        mins = rest // 60
        return f"{hours}h {mins}m" if mins else f"{hours}h"

    # ------------------------------- metadata --------------------------------

    def get_remote_file(self, repo_id: str, filename: str) -> RemoteFile:
        """Size, digest and pinned commit for one repository file."""
        key = (repo_id, filename)
        with self._remote_lock:
            cached = self._remote_cache.get(key)
        if cached is not None:
            return cached

        url = hf_hub_url(repo_id, filename)
        metadata = get_hf_file_metadata(url, timeout=float(self.config["metadata_timeout"]))
        size = getattr(metadata, "size", None)
        if not isinstance(size, int):
            raise DownloadError(f"{filename}: the repository did not report a file size")
        algorithm, digest = _digest_from_etag(getattr(metadata, "etag", None))
        remote = RemoteFile(
            repo_id,
            filename,
            size,
            algorithm,
            digest,
            str(getattr(metadata, "commit_hash", "") or "main"),
        )
        # The same response already carries the resolved CDN link, so the first
        # range request does not need another round trip to resolve it.
        location = str(getattr(metadata, "location", "") or url)
        self._remember_remote(
            remote,
            location,
            getattr(metadata, "xet_file_data", None),
        )
        return remote

    def _remember_remote(
        self,
        remote: RemoteFile,
        location: Optional[str] = None,
        xet_file_data=None,
    ) -> None:
        with self._remote_lock:
            self._remote_cache[remote.key] = remote
            file_hash = str(getattr(xet_file_data, "file_hash", "") or "")
            refresh_route = str(getattr(xet_file_data, "refresh_route", "") or "")
            if file_hash and refresh_route:
                self._xet_cache[remote.key] = (file_hash, refresh_route)
        if location:
            with self._url_lock:
                self._url_cache[remote.key] = (
                    location,
                    time.monotonic() + float(self.config["url_refresh_seconds"]),
                )

    def prefetch_metadata(self, requests_list: Sequence[Tuple[str, str]]) -> Dict[Tuple[str, str], RemoteFile]:
        """Resolve many files at once instead of one blocking call per file."""
        wanted = [key for key in dict.fromkeys(requests_list) if key not in self._remote_cache]
        if not wanted:
            return dict(self._remote_cache)

        by_repo: Dict[str, List[str]] = collections.defaultdict(list)
        for repo_id, filename in wanted:
            by_repo[repo_id].append(filename)

        # Repositories contributing several files are cheaper to list in one go.
        leftovers: List[Tuple[str, str]] = []
        for repo_id, filenames in by_repo.items():
            if len(filenames) < 3:
                leftovers.extend((repo_id, name) for name in filenames)
                continue
            try:
                _, index = repo_index(repo_id)
            except Exception as exc:
                logger.debug(f"Could not list {repo_id}: {exc}")
                leftovers.extend((repo_id, name) for name in filenames)
                continue
            for name in filenames:
                remote = index.get(name)
                if remote is None:
                    leftovers.append((repo_id, name))
                else:
                    self._remember_remote(remote)

        if leftovers:
            workers = max(1, min(int(self.config["metadata_workers"]), len(leftovers)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.get_remote_file, repo_id, name): (repo_id, name)
                    for repo_id, name in leftovers
                }
                for future in concurrent.futures.as_completed(futures):
                    with suppress(Exception):
                        future.result()
        return dict(self._remote_cache)

    # ------------------------------ verification -----------------------------

    def _verified_key(self, remote: RemoteFile) -> str:
        return f"{remote.repo_id}/{remote.filename}"

    def is_file_verified(self, remote: RemoteFile, filepath: Path) -> bool:
        """True when this exact file was verified earlier and never touched."""
        if not remote.digest:
            return False
        with self._cache_lock:
            record = self.verified_cache.get(self._verified_key(remote))
        if not isinstance(record, dict):
            return False
        try:
            stat = filepath.stat()
        except OSError:
            return False
        return (
            record.get("sha256") == remote.digest
            and record.get("size") == stat.st_size
            and abs(float(record.get("mtime", 0.0)) - stat.st_mtime) < 1.0
        )

    def mark_file_verified(self, remote: RemoteFile, filepath: Path) -> None:
        if not remote.digest:
            return
        try:
            stat = filepath.stat()
        except OSError:
            return
        with self._cache_lock:
            self.verified_cache[self._verified_key(remote)] = {
                "sha256": remote.digest,
                "algorithm": remote.digest_algorithm,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "verified_at": time.time(),
            }
            self._save_json_cache(self.verified_cache_file, self.verified_cache)

    def compute_digest(
        self,
        filepath: Path,
        algorithm: str,
        size: int,
        display: str,
        show_progress: bool = True,
    ) -> str:
        """Hash a file, showing progress because 15 GB is not instant."""
        if algorithm == "git-sha1":
            digest = hashlib.sha1()
            digest.update(b"blob %d\0" % size)
        else:
            digest = hashlib.new(algorithm)
        block = int(self.config["hash_block_size"])
        # Small files hash faster than a progress line is worth reading.
        show_progress = show_progress and size >= int(self.config["verify_progress_threshold"])
        read = 0
        started = time.monotonic()
        last_update = 0.0
        with open(filepath, "rb") as handle:
            while True:
                data = handle.read(block)
                if not data:
                    break
                digest.update(data)
                read += len(data)
                if not show_progress:
                    continue
                now = time.monotonic()
                if now - last_update >= 0.25:
                    percent = (read * 100.0 / size) if size else 100.0
                    speed = read / max(0.001, now - started)
                    self.show_progress_line(
                        f"[VERIFYING] {display}: {percent:.1f}% "
                        f"({self.format_bytes(read)}/{self.format_bytes(size)}) "
                        f"{self.format_bytes(speed)}/s"
                    )
                    last_update = now
        return digest.hexdigest().lower()

    # ------------------------------ URL handling -----------------------------

    def _headers_for(self, url: str) -> Dict[str, str]:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        if host == _HF_ENDPOINT_HOST:
            return dict(self._hf_headers)
        # Storage links are pre-signed. Forwarding the Hugging Face token to a
        # CDN host would leak it without granting any extra access.
        return {"user-agent": self._hf_headers.get("user-agent", "secourses-model-downloader")}

    def _download_target(self, remote: RemoteFile) -> Tuple[str, Dict[str, str]]:
        """The current CDN link for a file, refreshed before it can expire.

        Every range request asks for this, so the refresh happens outside the
        lock and only one thread does it. The others keep using the link they
        have, which still works: the refresh window is deliberately far shorter
        than the lifetime of a signed link.
        """
        with self._url_lock:
            cached = self._url_cache.get(remote.key)
            fresh = cached is not None and time.monotonic() < cached[1]
            if fresh or (cached is not None and remote.key in self._url_refreshing):
                return cached[0], self._headers_for(cached[0])
            self._url_refreshing.add(remote.key)

        url = hf_hub_url(remote.repo_id, remote.filename, revision=remote.commit_hash or "main")
        try:
            metadata = get_hf_file_metadata(url, timeout=float(self.config["metadata_timeout"]))
            location = str(getattr(metadata, "location", "") or url)
            self._remember_remote(
                remote,
                location,
                getattr(metadata, "xet_file_data", None),
            )
        except Exception as exc:
            logger.debug(f"Could not refresh the download link for {remote.filename}: {exc}")
            location = url
        with self._url_lock:
            self._url_cache[remote.key] = (
                location,
                time.monotonic() + float(self.config["url_refresh_seconds"]),
            )
            self._url_refreshing.discard(remote.key)
        return location, self._headers_for(location)

    def _expire_download_target(self, remote: RemoteFile) -> None:
        with self._url_lock:
            self._url_cache.pop(remote.key, None)

    @staticmethod
    def _status_code(exc: BaseException) -> Optional[int]:
        response = getattr(exc, "response", None)
        return getattr(response, "status_code", None)

    def _is_retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, (DownloadError, requests.exceptions.RequestException, OSError)):
            if isinstance(exc, RangeNotSupported):
                return False
            status = self._status_code(exc)
            if status is not None and status not in (408, 429) and status < 500:
                return False
            return True
        return False

    def _friendly_error(self, exc: BaseException) -> str:
        status = self._status_code(exc)
        if isinstance(exc, requests.exceptions.SSLError):
            return "TLS verification failed. Check the clock, CA certificates, proxy or firewall."
        if isinstance(exc, requests.exceptions.Timeout):
            return "The connection kept timing out. Re-run to resume from the saved bytes."
        if isinstance(exc, requests.exceptions.ConnectionError):
            return "Could not reach Hugging Face. Check the connection, proxy, DNS or firewall."
        if status == 401:
            return "Authentication failed. Run `hf auth login` if the repository is private."
        if status == 404:
            return "The file does not exist in the repository."
        if status == 429:
            return "Hugging Face rate-limited the request. Wait a moment, then re-run to resume."
        if status is not None and status >= 500:
            return f"Hugging Face returned HTTP {status}. Re-run later to resume."
        if isinstance(exc, PermissionError):
            return "The destination is locked. Close whatever is using the file and retry."
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
            return "Not enough free disk space for this download."
        return str(exc) or type(exc).__name__

    # ------------------------------ resume state -----------------------------

    @staticmethod
    def _staging_path(target: Path) -> Path:
        return Path(str(target) + ".part")

    @staticmethod
    def _state_path(target: Path) -> Path:
        return Path(str(target) + ".part.json")

    def _write_state(self, target: Path, remote: RemoteFile, ranges: _Ranges) -> None:
        payload = {
            "version": RESUME_STATE_VERSION,
            "repo_id": remote.repo_id,
            "filename": remote.filename,
            "commit_hash": remote.commit_hash,
            "digest": remote.digest,
            "size": remote.size,
            "ranges": ranges.snapshot(),
        }
        path = self._state_path(target)
        try:
            # Publish only ranges whose staged data has reached stable storage.
            # If power is lost after this point, recovery may conservatively
            # redownload newer bytes but cannot trust unwritten ones.
            staging = self._staging_path(target)
            if staging.is_file():
                descriptor = os.open(str(staging), os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            temp = Path(str(path) + ".tmp")
            with open(temp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            temp.replace(path)
            _fsync_parent(path)
        except OSError as exc:
            logger.debug(f"Could not save the resume map for {target.name}: {exc}")

    def _read_state(self, target: Path, remote: RemoteFile) -> Optional[_Ranges]:
        """Rebuild the saved byte map, rejecting anything that is not this file."""
        path = self._state_path(target)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            self.log(f"[WARNING] Ignoring an unreadable resume map for {target.name}")
            return None

        if not isinstance(data, dict):
            return None
        identity_changed = (
            data.get("repo_id") != remote.repo_id
            or data.get("filename") != remote.filename
            or data.get("size") != remote.size
            or data.get("digest") != remote.digest
            # A digest identifies content across commits. Without one, the
            # pinned commit is the only safe way to distinguish revisions.
            or (
                remote.digest is None
                and data.get("commit_hash") != remote.commit_hash
            )
        )
        if identity_changed:
            self.log(f"[WARNING] The saved ranges belong to another revision of {target.name}; starting over")
            return None

        try:
            on_disk = self._staging_path(target).stat().st_size
        except OSError:
            return None

        # Version 1 recorded one entry per connection as [start, end, done],
        # where only the first ``done`` bytes of that range had landed. Reading
        # it here is what lets a download interrupted by the previous release
        # continue instead of starting the file again.
        if data.get("version") == 1:
            entries = [
                (entry[0], entry[0] + entry[2])
                for entry in data.get("segments") or []
                if isinstance(entry, list) and len(entry) == 3 and all(isinstance(v, int) for v in entry)
            ]
        elif data.get("version") == RESUME_STATE_VERSION:
            entries = [
                (entry[0], entry[1])
                for entry in data.get("ranges") or []
                if isinstance(entry, list) and len(entry) == 2 and all(isinstance(v, int) for v in entry)
            ]
        else:
            return None

        # Anything past the end of the staging file was never actually written.
        clipped = [
            (start, min(end, remote.size, on_disk))
            for start, end in entries
            if 0 <= start < min(end, remote.size, on_disk)
        ]
        return _Ranges(clipped) if clipped else None

    def _discard_staging(self, target: Path) -> None:
        with suppress(OSError):
            self._staging_path(target).unlink()
        with suppress(OSError):
            self._state_path(target).unlink()
        with suppress(OSError):
            Path(str(self._state_path(target)) + ".tmp").unlink()
        # Numbered chunks written by releases before the single staging file.
        # iterdir rather than glob, because a file name is not a glob pattern.
        with suppress(OSError):
            for path in target.parent.iterdir():
                if path.name.startswith(target.name + ".part") and _LEGACY_PART_RE.search(path.name):
                    with suppress(OSError):
                        path.unlink()

    def _create_staging(self, staging: Path, size: int) -> None:
        """Make sure the staging file exists and is exactly ``size`` long.

        Reserving the length up front means every worker writes into a file that
        is already the right shape, so no write has to extend it. On Windows the
        file is also marked sparse, which keeps the reservation free instead of
        zero-filling gigabytes before the first byte arrives.
        """
        staging.parent.mkdir(parents=True, exist_ok=True)
        if not staging.exists():
            with open(staging, "wb"):
                pass
            _enable_sparse_file(staging)
        with open(staging, "r+b") as handle:
            if handle.seek(0, os.SEEK_END) != size:
                handle.truncate(size)
        _reserve_blocks(staging, size)

    # ------------------------------- transfer --------------------------------

    def _connection_count(self, pieces: int, file_size: int = 0) -> int:
        """Choose a bounded HTTP fleet from file size and machine resources."""
        requested = max(1, int(self.config["num_connections"]))
        if not int(self.config.get("http_adaptive_connections", 0)):
            return max(1, min(requested, pieces))

        floor = max(1, min(requested, 16))
        ceiling = max(floor, min(32, int(self.config.get("http_max_connections", 32))))
        memory = _total_memory_bytes()
        cpus = os.cpu_count() or 1
        # Low-resource systems remain at 16.  Bigger fleets are reserved for a
        # long queue, so fragmented resumes do not pay a large socket-startup
        # penalty merely because their ranges are scattered.
        machine_cap = floor
        if cpus >= 16 and memory >= 32 * 1024**3:
            machine_cap = min(24, ceiling)
        if cpus >= 24 and memory >= 64 * 1024**3:
            machine_cap = ceiling
        # The measured 48 GiB Hugging Face case was fastest at 16. Extra sockets
        # are useful mainly when a much larger object provides a long enough
        # queue to amortize handshakes, shaping waves, and connection rotation.
        if file_size and file_size < 64 * 1024**3:
            machine_cap = floor
        elif file_size and file_size < 128 * 1024**3:
            machine_cap = min(machine_cap, 24)
        if pieces < 512:
            machine_cap = floor
        elif pieces < 1024:
            machine_cap = min(machine_cap, 24)
        return max(1, min(machine_cap, pieces))

    def _piece_size_for(self, size: int) -> int:
        # Pieces grow with the file so a huge model keeps each request open
        # long enough to matter, without a lost piece ever costing much. The
        # byte map records what actually landed, so resuming does not depend
        # on the file being cut the same way twice.
        floor = max(1, int(self.config["piece_size"]))
        cap = max(floor, int(self.config["piece_size_max"]))
        target = max(1, int(self.config["pieces_per_file_target"]))
        return min(cap, max(floor, size // target))

    @staticmethod
    def _check_range_response(response, start: int, end: int, size: int) -> None:
        content_range = (response.headers.get("Content-Range") or "").strip()
        match = _CONTENT_RANGE_RE.fullmatch(content_range)
        if match is None:
            raise DownloadError(f"missing or invalid Content-Range: {content_range!r}")
        if tuple(int(value) for value in match.groups()) != (start, end, size):
            raise DownloadError(
                f"the server answered {content_range!r} instead of bytes {start}-{end}/{size}"
            )
        encoding = (response.headers.get("Content-Encoding") or "identity").lower()
        if encoding not in ("", "identity"):
            raise DownloadError(f"unexpected Content-Encoding {encoding!r} on a byte range")

    def _xet_details(self, remote: RemoteFile):
        """Return the optional native Xet backend and this file's Xet identity."""
        if (
            not int(self.config["xet_enabled"])
            or remote.size < int(self.config["xet_threshold"])
        ):
            return None
        try:
            import hf_xet

            if not all(
                hasattr(hf_xet, name)
                for name in ("XetConfig", "XetFileInfo", "XetSession")
            ):
                return None
            with self._remote_lock:
                cached = self._xet_cache.get(remote.key)
            if cached is None:
                url = hf_hub_url(
                    remote.repo_id,
                    remote.filename,
                    revision=remote.commit_hash or "main",
                )
                metadata = get_hf_file_metadata(
                    url,
                    timeout=float(self.config["metadata_timeout"]),
                )
                if getattr(metadata, "size", None) != remote.size:
                    raise DownloadError("Xet metadata reported a different file size")
                _, digest = _digest_from_etag(getattr(metadata, "etag", None))
                if remote.digest and digest != remote.digest:
                    raise DownloadError("Xet metadata belongs to a different file revision")
                self._remember_remote(
                    remote,
                    str(getattr(metadata, "location", "") or url),
                    getattr(metadata, "xet_file_data", None),
                )
                with self._remote_lock:
                    cached = self._xet_cache.get(remote.key)
            if cached is None:
                return None
            return hf_xet, cached[0], cached[1]
        except Exception as exc:
            logger.debug(f"Native Xet is unavailable for {remote.filename}: {exc}")
            return None

    def _new_xet_session(self, hf_xet):
        """Build the measured fast Xet profile without overcommitting small PCs."""
        concurrency = max(1, min(124, int(self.config["xet_concurrency"])))
        override = os.environ.get("HF_XET_FIXED_DOWNLOAD_CONCURRENCY")
        if override:
            with suppress(ValueError):
                concurrency = max(1, min(124, int(override)))

        total_memory = _total_memory_bytes()
        if total_memory and total_memory < 8 * 1024**3:
            concurrency = min(concurrency, 8)
        elif total_memory and total_memory < 16 * 1024**3:
            concurrency = min(concurrency, 12)
        elif total_memory and total_memory < 32 * 1024**3:
            concurrency = min(concurrency, 16)
        elif total_memory and total_memory < int(self.config["xet_high_memory_threshold"]):
            concurrency = min(concurrency, 24)

        updates = {
            "client.enable_adaptive_concurrency": False,
            "client.ac_initial_download_concurrency": concurrency,
            "client.ac_min_download_concurrency": concurrency,
            "client.ac_max_download_concurrency": concurrency,
            "client.max_idle_connections": max(16, concurrency),
        }
        gib = 1024**3
        mib = 1024**2
        if total_memory and total_memory < 8 * gib:
            updates.update(
                {
                    "reconstruction.min_reconstruction_fetch_size": 64 * mib,
                    "reconstruction.max_reconstruction_fetch_size": 512 * mib,
                    "reconstruction.download_buffer_size": 256 * mib,
                    "reconstruction.download_buffer_perfile_size": 128 * mib,
                    "reconstruction.download_buffer_limit": 512 * mib,
                    "reconstruction.min_prefetch_buffer": 128 * mib,
                }
            )
        elif total_memory and total_memory < 16 * gib:
            updates.update(
                {
                    "reconstruction.min_reconstruction_fetch_size": 128 * mib,
                    "reconstruction.max_reconstruction_fetch_size": 1 * gib,
                    "reconstruction.download_buffer_size": 512 * mib,
                    "reconstruction.download_buffer_perfile_size": 256 * mib,
                    "reconstruction.download_buffer_limit": 1 * gib,
                    "reconstruction.min_prefetch_buffer": 256 * mib,
                }
            )
        elif total_memory and total_memory < 32 * gib:
            updates.update(
                {
                    "reconstruction.max_reconstruction_fetch_size": 2 * gib,
                    "reconstruction.download_buffer_size": 1 * gib,
                    "reconstruction.download_buffer_perfile_size": 256 * mib,
                    "reconstruction.download_buffer_limit": 2 * gib,
                    "reconstruction.min_prefetch_buffer": 512 * mib,
                }
            )
        # Keep hf_xet's bounded reconstruction-buffer defaults on every host.
        # Larger buffers increased peak RSS roughly in proportion to model size
        # (several GiB for a single 20 GiB model) without being portable to the
        # ordinary Windows and Linux computers this downloader targets.
        config = hf_xet.XetConfig().with_config(updates)
        return hf_xet.XetSession(config), concurrency

    def _run_xet(
        self,
        remote: RemoteFile,
        target: Path,
        display: str,
        ranges: _Ranges,
        gaps: Sequence[Tuple[int, int]],
        details,
        show_progress: bool,
        shared_meter: Optional[_SpeedMeter],
    ) -> None:
        """Stream exact missing ranges through native Xet and retain our map."""
        hf_xet, file_hash, refresh_route = details
        session, concurrency = self._new_xet_session(hf_xet)
        info = hf_xet.XetFileInfo(file_hash, remote.size)
        group = session.new_download_stream_group(
            token_refresh_url=refresh_route,
            token_refresh_headers=dict(self._hf_headers),
            custom_headers={
                "user-agent": self._hf_headers.get(
                    "user-agent", "secourses-model-downloader"
                )
            },
        )
        staging = self._staging_path(target)
        meter = _SpeedMeter(float(self.config["speed_window"]))
        pending: List[Tuple[int, int]] = []
        pending_bytes = 0
        flush_size = max(1, int(self.config["xet_range_flush_size"]))
        already_done = ranges.total
        action = "RESUMING" if already_done else "DOWNLOADING"
        started = time.monotonic()
        last_state = started
        last_progress = 0.0

        def commit_ranges() -> None:
            nonlocal pending, pending_bytes
            if pending:
                ranges.add_many(pending)
                pending = []
                pending_bytes = 0

        if show_progress:
            detail = self.format_bytes(remote.size)
            if already_done:
                detail += f", {self.format_bytes(already_done)} already on disk"
            self.log(
                f"[{action}] {display} ({detail}) with native Xet "
                f"({concurrency} streams)"
            )

        try:
            with open(staging, "r+b", buffering=0) as handle:
                for gap_start, gap_stop in gaps:
                    stream = group.download_unordered_stream(
                        info,
                        start=gap_start,
                        end=gap_stop,
                    )
                    try:
                        for relative, data in stream:
                            relative = int(relative)
                            absolute = gap_start + relative
                            stop = absolute + len(data)
                            if (
                                not data
                                or relative < 0
                                or absolute < gap_start
                                or stop > gap_stop
                            ):
                                raise DownloadError("Xet returned an invalid byte range")
                            handle.seek(absolute)
                            view = memoryview(data)
                            written = 0
                            while written < len(view):
                                step = handle.write(view[written:])
                                if not step:
                                    raise DownloadError(
                                        "the destination stopped accepting Xet writes"
                                    )
                                written += step
                            pending.append((absolute, stop))
                            pending_bytes += len(data)
                            meter.add(len(data))

                            now = time.monotonic()
                            if pending_bytes >= flush_size:
                                commit_ranges()
                            if now - last_state >= float(self.config["state_flush_interval"]):
                                commit_ranges()
                                self._write_state(target, remote, ranges)
                                last_state = now
                            if (
                                show_progress
                                and now - last_progress >= float(self.config["progress_interval"])
                            ):
                                current = min(remote.size, ranges.total + pending_bytes)
                                speed = meter.rate()
                                remaining = max(0, remote.size - current)
                                eta = self.format_time(remaining / speed) if speed > 0 else "--"
                                self.show_progress_line(
                                    f"[{action}] {display}: "
                                    f"{current * 100.0 / max(1, remote.size):.1f}% "
                                    f"({self.format_bytes(current)}/{self.format_bytes(remote.size)}) "
                                    f"{self.format_bytes(speed)}/s ETA {eta} | Xet {concurrency} streams"
                                )
                                last_progress = now
                    finally:
                        with suppress(Exception):
                            stream.cancel()
            commit_ranges()
            if ranges.total != remote.size:
                raise DownloadError(
                    f"{self.format_bytes(remote.size - ranges.total)} are still missing"
                )
        except KeyboardInterrupt:
            with suppress(Exception):
                session.sigint_abort()
            raise
        except DownloadError:
            raise
        except Exception as exc:
            raise DownloadError(f"native Xet transfer failed: {exc}") from exc
        finally:
            commit_ranges()
            self._write_state(target, remote, ranges)
            if shared_meter is not None:
                shared_meter.add(meter.total)
            if show_progress:
                self.finalize_progress_line()

        elapsed = max(0.001, time.monotonic() - started)
        if show_progress and meter.total:
            self.log(
                f"[TRANSFERRED] {display}: {self.format_bytes(meter.total)} in "
                f"{self.format_time(elapsed)} ({self.format_bytes(meter.total / elapsed)}/s)"
            )

    def _try_run_xet(
        self,
        remote: RemoteFile,
        target: Path,
        display: str,
        ranges: _Ranges,
        show_progress: bool,
        shared_meter: Optional[_SpeedMeter],
    ) -> bool:
        details = self._xet_details(remote)
        if details is None:
            return False
        gaps = ranges.missing(0, remote.size)
        if not gaps:
            return False
        configured_limit = int(self.config.get("xet_max_resume_gaps", 0))
        if show_progress and ranges.total and len(gaps) > 1:
            suffix = ""
            if configured_limit > 0 and len(gaps) > configured_limit:
                suffix = f" (legacy limit {configured_limit} ignored safely)"
            self.log(
                f"[RESUME] {display}: native Xet will fetch "
                f"{len(gaps)} exact missing ranges{suffix}"
            )
        self._run_xet(
            remote,
            target,
            display,
            ranges,
            gaps,
            details,
            show_progress,
            shared_meter,
        )
        return True

    def _fetch_piece(
        self,
        remote: RemoteFile,
        link: _Link,
        handle,
        ranges: _Ranges,
        start: int,
        stop: int,
        cancel: threading.Event,
        meter: _SpeedMeter,
        progress: _Progress,
    ) -> None:
        """Fetch ``[start, stop)`` and record every byte that reaches the file.

        Whatever arrives before a failure is written and kept, so a retry asks
        only for the bytes that are genuinely still missing, whether the retry
        happens seconds later or on a re-run days later.
        """
        block = int(self.config["read_block_size"])
        flush_at = int(self.config["write_block_size"])
        timeout = (float(self.config["connect_timeout"]), float(self.config["read_timeout"]))
        min_rate = float(self.config["min_piece_rate"])
        min_seconds = float(self.config["min_piece_seconds"])
        slow_grace = float(self.config["slow_connection_grace"])
        slow_fraction = float(self.config["slow_connection_fraction"])
        max_retries = max(1, int(self.config["max_retries"]))
        buffer = bytearray()
        offset = start
        # ``stuck`` counts only the failures that moved nothing. A connection
        # that dies after delivering part of its range has still made the file
        # smaller, so it earns a fresh budget; the hard cap is what guarantees
        # the loop ends even against a server that fails in tiny increments.
        stuck = 0
        attempts = 0

        def flush() -> None:
            """Put the batch on disk, then record those bytes as present."""
            nonlocal buffer, offset
            moved = len(buffer)
            if not moved:
                return
            with memoryview(buffer) as view:
                handle.seek(offset)
                written = 0
                while written < moved:
                    step = handle.write(view[written:])
                    if not step:
                        raise DownloadError("the destination stopped accepting writes")
                    written += step
            # Only bytes that survived the write are claimed, so a crash between
            # these two lines costs a retry rather than a corrupt file.
            ranges.add(offset, offset + moved)
            offset += moved
            buffer = bytearray()
            meter.add(moved)
            progress.touch()

        while offset < stop:
            if cancel.is_set():
                raise _Cancelled()
            url, base_headers = self._download_target(remote)
            headers = dict(base_headers)
            headers["Range"] = f"bytes={offset}-{stop - 1}"
            headers["Accept-Encoding"] = "identity"
            reached = offset
            try:
                with link.session().get(
                    url, headers=headers, stream=True, timeout=timeout
                ) as response:
                    if response.status_code in (401, 403, 410):
                        self._expire_download_target(remote)
                        raise DownloadError(f"the download link expired (HTTP {response.status_code})")
                    if response.status_code == 200:
                        raise RangeNotSupported("the server ignored the Range header")
                    if response.status_code != 206:
                        response.raise_for_status()
                        raise DownloadError(f"unexpected HTTP status {response.status_code}")
                    self._check_range_response(response, offset, stop - 1, remote.size)

                    raw = response.raw
                    began = time.monotonic()
                    received = 0
                    while offset + len(buffer) < stop:
                        if cancel.is_set():
                            raise _Cancelled()
                        data = raw.read(min(block, stop - offset - len(buffer)))
                        if not data:
                            break
                        buffer += data
                        received += len(data)
                        if len(buffer) >= flush_at:
                            flush()
                        elapsed = time.monotonic() - began
                        # A connection that is still crawling after a fair
                        # chance is replaced instead of holding the file back:
                        # below an absolute floor, or far behind what the
                        # other connections are moving right now.
                        if elapsed >= min_seconds and received / elapsed < min_rate:
                            raise DownloadError(
                                f"this connection settled at "
                                f"{self.format_bytes(received / elapsed)}/s"
                            )
                        if elapsed >= slow_grace:
                            fleet = meter.rate() / max(1, progress.active)
                            if (
                                fleet * slow_fraction > min_rate
                                and received / elapsed < fleet * slow_fraction
                            ):
                                raise _TooSlow(
                                    f"{self.format_bytes(received / elapsed)}/s "
                                    f"against a fleet average of "
                                    f"{self.format_bytes(fleet)}/s"
                                )

                flush()
                if offset >= stop:
                    return
                raise DownloadError("the connection closed before the range was complete")
            except (_Cancelled, RangeNotSupported):
                raise
            except Exception as exc:
                # Whatever went wrong, that socket has had its chance.
                link.drop()
                if cancel.is_set():
                    raise _Cancelled() from exc
                # Bytes already in hand are still good; only the rest is retried.
                with suppress(Exception):
                    flush()
                buffer = bytearray()
                attempts += 1
                stuck = 0 if offset > reached else stuck + 1
                if (
                    stuck >= max_retries
                    or attempts >= max_retries * 8
                    or not self._is_retryable(exc)
                ):
                    raise DownloadError(
                        f"bytes {offset}-{stop - 1}: {self._friendly_error(exc)}"
                    ) from exc
                if isinstance(exc, _TooSlow):
                    # The path was slow, not broken; dial again without a pause.
                    continue
                delay = min(
                    float(self.config["retry_delay"]) * (2 ** max(0, stuck - 1)),
                    float(self.config["max_retry_delay"]),
                )
                if cancel.wait(delay):
                    raise _Cancelled()

    def _run_pieces(
        self,
        remote: RemoteFile,
        target: Path,
        display: str,
        ranges: _Ranges,
        show_progress: bool,
        shared_meter: Optional[_SpeedMeter] = None,
    ) -> None:
        """Fetch every missing piece of one file over several connections."""
        staging = self._staging_path(target)
        self._create_staging(staging, remote.size)

        if self._try_run_xet(
            remote, target, display, ranges, show_progress, shared_meter
        ):
            return

        piece = self._piece_size_for(remote.size)
        work: List[Tuple[int, int]] = []
        for piece_start in range(0, remote.size, piece):
            piece_stop = min(piece_start + piece, remote.size)
            # Asking only for genuine gaps preserves every byte from an older
            # run even when this release uses a different piece size.
            work.extend(ranges.missing(piece_start, piece_stop))
        workers = self._connection_count(len(work), remote.size)
        per_session = max(1, int(self.config["pieces_per_connection"]))
        max_age = float(self.config["max_connection_seconds"])
        age_jitter = float(self.config["connection_age_jitter"])
        meter = _SpeedMeter(float(self.config["speed_window"]))
        cancel = threading.Event()
        progress = _Progress()
        errors: List[BaseException] = []
        errors_lock = threading.Lock()
        counter = [0]
        counter_lock = threading.Lock()
        already_done = ranges.total

        def claim() -> Optional[Tuple[int, int]]:
            with counter_lock:
                index = counter[0]
                if index >= len(work):
                    return None
                counter[0] = index + 1
                return work[index]

        action = "RESUMING" if already_done else "DOWNLOADING"
        if show_progress:
            detail = self.format_bytes(remote.size)
            if already_done:
                detail += f", {self.format_bytes(already_done)} already on disk"
            self.log(f"[{action}] {display} ({detail}) over {workers} connections")

        def worker() -> None:
            # Pieces are claimed in file order, so the connections stay clustered
            # around one moving point in the file rather than writing to a dozen
            # places gigabytes apart.
            link = _Link(self._create_session, per_session, max_age, age_jitter)
            try:
                with open(staging, "r+b", buffering=0) as handle:
                    while not cancel.is_set():
                        claimed = claim()
                        if claimed is None:
                            return
                        start, stop = claimed
                        progress.enter()
                        try:
                            self._fetch_piece(
                                remote, link, handle, ranges,
                                start, stop, cancel, meter, progress,
                            )
                        finally:
                            progress.leave()
            except _Cancelled:
                pass
            except BaseException as exc:  # noqa: BLE001 - reported to the caller
                cancel.set()
                with errors_lock:
                    errors.append(exc)
            finally:
                link.drop()

        started = time.monotonic()
        stall_timeout = float(self.config["stall_timeout"])
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="hf-piece"
        )
        interrupted: Optional[BaseException] = None
        try:
            pending = {executor.submit(worker) for _ in range(workers)}
            last_state = time.monotonic()
            while pending:
                _, pending = concurrent.futures.wait(
                    pending, timeout=float(self.config["progress_interval"])
                )
                now = time.monotonic()
                current = ranges.total
                if show_progress:
                    speed = meter.rate()
                    remaining = max(0, remote.size - current)
                    eta = self.format_time(remaining / speed) if speed > 0 else "--"
                    self.show_progress_line(
                        f"[{action}] {display}: {current * 100.0 / max(1, remote.size):.1f}% "
                        f"({self.format_bytes(current)}/{self.format_bytes(remote.size)}) "
                        f"{self.format_bytes(speed)}/s ETA {eta} | {progress.active} conns "
                        f"@ {self.format_bytes(speed / max(1, progress.active))}/s"
                    )
                if now - last_state >= float(self.config["state_flush_interval"]):
                    self._write_state(target, remote, ranges)
                    last_state = now
                if not cancel.is_set() and now - progress.last > stall_timeout:
                    # Nothing has reached the file for minutes. Stop the workers
                    # and let the caller resume rather than sit here forever.
                    cancel.set()
                    with errors_lock:
                        errors.append(
                            DownloadError(
                                f"nothing arrived for {self.format_time(stall_timeout)}"
                            )
                        )
        except BaseException as exc:  # KeyboardInterrupt included on purpose
            cancel.set()
            interrupted = exc
        finally:
            executor.shutdown(wait=True)
            self._write_state(target, remote, ranges)
            if show_progress:
                self.finalize_progress_line()

        if shared_meter is not None:
            shared_meter.add(meter.total)
        if interrupted is not None:
            raise interrupted
        if errors:
            # A server that ignores Range says so on every connection at once,
            # and that answer decides the fallback, so it outranks the rest.
            raise next(
                (exc for exc in errors if isinstance(exc, RangeNotSupported)), errors[0]
            )

        if ranges.total != remote.size:
            raise DownloadError(
                f"{self.format_bytes(remote.size - ranges.total)} are still missing"
            )
        elapsed = max(0.001, time.monotonic() - started)
        if show_progress and meter.total:
            self.log(
                f"[TRANSFERRED] {display}: {self.format_bytes(meter.total)} in "
                f"{self.format_time(elapsed)} ({self.format_bytes(meter.total / elapsed)}/s)"
            )

    def _stream_whole_file(
        self,
        remote: RemoteFile,
        target: Path,
        display: str,
        show_progress: bool,
    ) -> None:
        """One-connection fallback for servers that refuse byte ranges."""
        staging = self._staging_path(target)
        self._discard_staging(target)
        self._create_staging(staging, 0)
        timeout = (float(self.config["connect_timeout"]), float(self.config["read_timeout"]))
        url, headers = self._download_target(remote)
        headers = dict(headers)
        headers["Accept-Encoding"] = "identity"
        meter = _SpeedMeter(float(self.config["speed_window"]))
        started = time.monotonic()
        # A fallback can run alongside other small-file downloads. Give it a
        # private Session: requests.Session is not safe to mutate/use from many
        # threads at once.
        with self._create_session(1) as session:
            with session.get(url, headers=headers, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(staging, "wb") as handle:
                    for data in response.iter_content(chunk_size=int(self.config["read_block_size"])):
                        if not data:
                            continue
                        handle.write(data)
                        meter.add(len(data))
                        if show_progress:
                            current = meter.total
                            speed = meter.rate()
                            self.show_progress_line(
                                f"[DOWNLOADING] {display}: {current * 100.0 / max(1, remote.size):.1f}% "
                                f"({self.format_bytes(current)}/{self.format_bytes(remote.size)}) "
                                f"{self.format_bytes(speed)}/s"
                            )
        if show_progress:
            elapsed = max(0.001, time.monotonic() - started)
            self.finalize_progress_line(
                f"[TRANSFERRED] {display}: {self.format_bytes(meter.total)} in "
                f"{self.format_time(elapsed)}"
            )

    def _check_free_space(self, target: Path, size: int, display: str) -> None:
        """Refuse a transfer that cannot possibly fit, before any bytes move."""
        try:
            # Space the staging file already holds is space this does not need
            # again, which is what keeps a resume from being refused on a disk
            # that is nearly full precisely because the download is on it.
            staged = _allocated_file_bytes(self._staging_path(target))
        except OSError:
            staged = 0
        needed = size - staged
        if needed <= 0:
            return
        try:
            free = shutil.disk_usage(target.parent).free
        except OSError:
            return
        if free < needed + (64 << 20):
            raise DownloadError(
                f"{display} needs {self.format_bytes(needed)} but only "
                f"{self.format_bytes(free)} is free on {target.parent}"
            )

    def _install(self, staging: Path, target: Path) -> None:
        attempts = 5
        for attempt in range(attempts):
            try:
                os.replace(staging, target)
                _fsync_parent(target)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                delay = min(0.25 * (2 ** attempt), 2.0)
                self.log(f"[RETRY] {target.name} is locked; replacing it again in {delay:.2f}s")
                time.sleep(delay)

    def _already_complete(self, remote: RemoteFile, target: Path, display: str) -> bool:
        if not target.is_file():
            return False
        try:
            stat = target.stat()
        except OSError:
            return False
        if stat.st_size != remote.size:
            return False
        if self.skip_verify or not remote.digest:
            self.log(f"[SKIP] {display} is already complete ({self.format_bytes(remote.size)})")
            return True
        if self.is_file_verified(remote, target):
            self.log(f"[SKIP] {display} is already complete and verified")
            return True
        actual = self.compute_digest(target, remote.digest_algorithm, remote.size, display)
        if actual == remote.digest:
            self.finalize_progress_line(f"[SKIP] {display} is already complete and verified")
            self.mark_file_verified(remote, target)
            return True
        self.finalize_progress_line(
            f"[WARNING] {display} does not match the repository copy and is being downloaded again"
        )
        return False

    def ensure_file(
        self,
        remote: RemoteFile,
        target: Path,
        *,
        show_progress: bool = True,
        shared_meter: Optional[_SpeedMeter] = None,
        display: Optional[str] = None,
        retry_corrupt: bool = True,
    ) -> bool:
        """Download, verify and install one file, resuming whatever exists."""
        # Repositories repeat names such as flownet.pkl across version folders,
        # so the caller can pass the path that tells them apart.
        display = display or target.name

        try:
            target.parent.mkdir(parents=True, exist_ok=True)

            if self._already_complete(remote, target, display):
                self._discard_staging(target)
                return True

            ranges = self._read_state(target, remote)
            if ranges is None:
                self._discard_staging(target)
                ranges = _Ranges()
            self._check_free_space(target, remote.size, display)

            # One transfer normally covers the whole file. When the network
            # disappears underneath it, the byte map is simply picked back up:
            # only a restart that gains nothing at all counts against the
            # budget, so a bad line costs time rather than the download.
            budget = max(1, int(self.config["file_attempts"]))
            stuck = 0
            attempts = 0
            refusals = 0
            unranged = False
            while True:
                reached = ranges.total
                try:
                    if unranged:
                        self._stream_whole_file(remote, target, display, show_progress)
                    else:
                        self._run_pieces(
                            remote, target, display, ranges, show_progress, shared_meter
                        )
                    break
                except RangeNotSupported:
                    # Falling back means restarting the file on one connection,
                    # so a CDN that answers one request with 200 instead of 206
                    # must not be able to throw away gigabytes on its own.
                    refusals = 0 if ranges.total > reached else refusals + 1
                    if ranges.total and refusals < 2:
                        self.log(
                            f"[RETRY] {display}: a range request came back unranged; "
                            f"trying again before giving up {self.format_bytes(ranges.total)}"
                        )
                        time.sleep(5.0)
                        continue
                    self.log(f"[WARNING] {display}: this server does not serve byte ranges")
                    # The single-stream fallback runs through the same retry
                    # budget below, so a drop at 14 GB is not the end of it.
                    unranged = True
                    continue
                except (DownloadError, requests.exceptions.RequestException, OSError) as exc:
                    attempts += 1
                    stuck = 0 if ranges.total > reached else stuck + 1
                    if stuck >= budget or attempts >= budget * 8:
                        raise
                    delay = min(5.0 * stuck, 30.0)
                    self.log(
                        f"[RETRY] {display}: {self._friendly_error(exc)}; "
                        f"resuming from {self.format_bytes(ranges.total)} "
                        f"({ranges.total * 100.0 / max(1, remote.size):.1f}%) in {delay:.0f}s"
                    )
                    time.sleep(delay)

            staging = self._staging_path(target)
            actual_size = staging.stat().st_size
            if actual_size != remote.size:
                raise DownloadError(
                    f"the staged file is {self.format_bytes(actual_size)} but the repository "
                    f"reports {self.format_bytes(remote.size)}"
                )

            if not self.skip_verify and remote.digest:
                actual = self.compute_digest(staging, remote.digest_algorithm, remote.size, display)
                if actual != remote.digest:
                    self.finalize_progress_line(
                        f"[ERROR] {display} failed verification and was discarded"
                    )
                    self.log(f"  Expected: {remote.digest}")
                    self.log(f"  Got:      {actual}")
                    self._discard_staging(target)
                    if retry_corrupt:
                        self.log(f"[RETRY] {display}: downloading it again from scratch")
                        return self.ensure_file(
                            remote,
                            target,
                            show_progress=show_progress,
                            shared_meter=shared_meter,
                            display=display,
                            retry_corrupt=False,
                        )
                    return False
                self.finalize_progress_line(f"[VERIFIED] {display}: {actual[:16]}...")
            elif not remote.digest:
                self.log(f"[INFO] {display}: the repository publishes no digest, size was checked")

            self._install(staging, target)
            with suppress(OSError):
                self._state_path(target).unlink()
            self.mark_file_verified(remote, target)
            self.log(f"[OK] {display} is ready ({self.format_bytes(remote.size)})")
            return True

        except KeyboardInterrupt:
            self.finalize_progress_line()
            self.log(f"[INTERRUPTED] {display}: the transferred bytes were kept for the next run")
            raise
        except Exception as exc:
            self.finalize_progress_line()
            self.log(f"[FAILED] {display}: {self._friendly_error(exc)}")
            self.log("[RESUME] The bytes already fetched were kept; re-run to continue from there.")
            return False

    # -------------------------------- public API -----------------------------

    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: Path,
        local_filename: Optional[str] = None,
    ) -> bool:
        """Fetch one repository file into ``local_dir``."""
        target = Path(local_dir) / (local_filename if local_filename is not None else filename)
        try:
            remote = self.get_remote_file(repo_id, filename)
        except Exception as exc:
            self.log(f"[FAILED] {filename}: {self._friendly_error(exc)}")
            return False
        return self.ensure_file(remote, target)

    def download_many(
        self,
        items: Sequence[Tuple[str, str, Path, str]],
    ) -> Tuple[int, int, List[str]]:
        """Fetch a batch of files, small ones several at a time.

        Repositories such as RIFE or BestImageUpscalers are dozens of modest
        files where connection setup, not bandwidth, dominates, so those run
        concurrently. Large weights still get every connection to themselves.
        """
        if not items:
            return 0, 0, []

        self.prefetch_metadata([(repo_id, name) for repo_id, name, _, _ in items])

        resolved: List[Tuple[RemoteFile, Path, str]] = []
        failed: List[str] = []
        for repo_id, remote_name, target_dir, local_name in items:
            try:
                remote = self.get_remote_file(repo_id, remote_name)
            except Exception as exc:
                self.log(f"[FAILED] {remote_name}: {self._friendly_error(exc)}")
                failed.append(remote_name)
                continue
            resolved.append((remote, Path(target_dir) / local_name, str(local_name)))

        threshold = int(self.config["parallel_threshold"])
        small = [entry for entry in resolved if entry[0].size < threshold]
        large = [entry for entry in resolved if entry[0].size >= threshold]
        successful = 0
        total = len(resolved)
        index = 0

        if small:
            workers = max(1, min(int(self.config["small_file_workers"]), len(small)))
            self.log(f"[BATCH] Fetching {len(small)} smaller file(s) over {workers} parallel downloads")
            meter = _SpeedMeter(float(self.config["speed_window"]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self.ensure_file,
                        remote,
                        target,
                        show_progress=False,
                        shared_meter=meter,
                        display=label,
                    ): label
                    for remote, target, label in small
                }
                for future in concurrent.futures.as_completed(futures):
                    label = futures[future]
                    index += 1
                    try:
                        ok = future.result()
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        self.log(f"[FAILED] {label}: {self._friendly_error(exc)}")
                        ok = False
                    if ok:
                        successful += 1
                    else:
                        failed.append(label)
            elapsed_note = self.format_bytes(meter.total)
            self.log(f"[BATCH] {len(small)} smaller file(s) done ({elapsed_note} transferred)")

        for remote, target, label in large:
            index += 1
            self.log(f"\n[{index}/{total}] {label} ({self.format_bytes(remote.size)})")
            if self.ensure_file(remote, target, display=label):
                successful += 1
            else:
                failed.append(label)

        return successful, len(failed), failed

    def download_repo(
        self,
        repo_id: str,
        local_dir: Path,
        exclude_patterns: Optional[List[str]] = None,
        include_prefixes: Optional[List[str]] = None,
        strip_prefix: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Download every matching file from a repository.

        Args:
            repo_id: Repository such as ``MonsterMMORPG/Wan_GGUF``.
            local_dir: Directory that receives the files.
            exclude_patterns: Substrings/suffixes that skip a file.
            include_prefixes: Optional repository paths to restrict the download to.
            strip_prefix: Optional repository prefix removed from local paths.

        Returns:
            ``(successful_count, failed_count)``
        """
        if exclude_patterns is None:
            exclude_patterns = [".gitattributes", ".gitignore"]

        prefixes = [value.strip("/") for value in include_prefixes or [] if value and value.strip("/")]
        strip = strip_prefix.strip("/") if strip_prefix and strip_prefix.strip("/") else None

        self.log(f"[INFO] Listing files in repository: {repo_id}")
        try:
            names = list_repo_file_names(repo_id)
        except Exception as exc:
            self.log(f"[ERROR] Could not list the repository: {self._friendly_error(exc)}")
            return 0, 1

        items: List[Tuple[str, str, Path, str]] = []
        for name in names:
            if any(pattern in name or name.endswith(pattern) for pattern in exclude_patterns):
                continue
            if prefixes and not any(name == prefix or name.startswith(prefix + "/") for prefix in prefixes):
                continue
            local_name = name
            if strip:
                token = strip + "/"
                if name.startswith(token):
                    local_name = name[len(token):]
                elif name == strip:
                    continue
            items.append((repo_id, name, Path(local_dir), local_name))

        self.log(f"[INFO] Found {len(items)} files to download")
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        successful, failed_count, failed_files = self.download_many(items)
        if failed_files:
            self.log(f"\n[WARNING] Failed to download {len(failed_files)} file(s):")
            for name in failed_files:
                self.log(f"  - {name}")
        return successful, failed_count


def _repo_file_items(
    repo_id: str,
    target_dir: Path,
    *,
    include_prefix: Optional[str] = None,
    strip_prefix: Optional[str] = None,
    exclude_files: Optional[List[str]] = None,
) -> List[Tuple[str, str, Path, str]]:
    """Resolve a repository folder into explicit downloads for one selected model."""
    print(f"[INFO] Listing required files in {repo_id}", flush=True)
    files = list_repo_file_names(repo_id)
    prefix = str(include_prefix or "").strip("/")
    strip = str(strip_prefix or "").strip("/")
    excluded = set(exclude_files or [])
    items: List[Tuple[str, str, Path, str]] = []
    for remote_name in files:
        if prefix and not (remote_name == prefix or remote_name.startswith(prefix + "/")):
            continue
        if remote_name in excluded:
            continue
        if "/__pycache__/" in f"/{remote_name}" or remote_name.endswith((".pyc", ".DS_Store")):
            continue
        if remote_name.endswith((".md", ".png", "/LICENSE")) or remote_name in {".gitattributes", ".gitignore", "LICENSE"}:
            continue
        local_name = remote_name
        if strip:
            token = strip + "/"
            if remote_name.startswith(token):
                local_name = remote_name[len(token):]
            elif remote_name == strip:
                continue
        items.append((repo_id, remote_name, target_dir, local_name))
    return items


def _download_selected_items(
    title: str,
    items: List[Tuple[str, str, Path, str]],
    *,
    skip_verify: bool = False,
    dry_run: bool = False,
) -> bool:
    if not items:
        print(f"[ERROR] No downloadable files were found for {title}", flush=True)
        return False
    print("\n" + "=" * 60, flush=True)
    print(title, flush=True)
    print("=" * 60, flush=True)
    for repo_id, remote_name, target_dir, local_name in items:
        print(f"  {repo_id}/{remote_name} -> {target_dir / local_name}", flush=True)
    if dry_run:
        print("[DRY RUN] No files downloaded.", flush=True)
        return True

    downloader = RobustDownloader(DOWNLOAD_CONFIG, skip_verify=skip_verify)
    _, failed_count, failed_names = downloader.download_many(items)
    if failed_count:
        for name in failed_names:
            print(f"[ERROR] Could not prepare {name}", flush=True)
        return False
    print(f"\n[READY] {title}", flush=True)
    return True


def ensure_seedvr2_model(
    model_filename: str,
    *,
    int8_convrot: bool = False,
    skip_verify: bool = False,
    dry_run: bool = False,
) -> bool:
    model_filename = Path(str(model_filename or "")).name
    config_by_filename = {
        config["filename"]: config for config in SEEDVR2_MODEL_CONFIGS.values()
    }
    items: List[Tuple[str, str, Path, str]] = []

    if int8_convrot:
        cache_filename = PREBUILT_INT8_CONVROT_FILES.get(model_filename)
        if not cache_filename:
            print(
                f"[ERROR] INT8 ConvRot is not available for SeedVR2 model: {model_filename}",
                flush=True,
            )
            return False
        items.append((SEEDVR2_REPO_ID, cache_filename, SEEDVR2_MODELS_DIR, cache_filename))
    else:
        config = config_by_filename.get(model_filename)
        if config is None and model_filename.lower().endswith(".gguf"):
            config = {"filename": model_filename, "repo_id": SEEDVR2_GGUF_REPO_ID}
        if config is None:
            print(f"[ERROR] No automatic download is configured for SeedVR2 model: {model_filename}", flush=True)
            return False
        repo_id = config.get("repo_id", SEEDVR2_REPO_ID)
        items.append((repo_id, model_filename, SEEDVR2_MODELS_DIR, model_filename))

    vae_name = SEEDVR2_MODEL_CONFIGS["vae"]["filename"]
    if model_filename != vae_name:
        items.append((SEEDVR2_REPO_ID, vae_name, SEEDVR2_MODELS_DIR, vae_name))
    return _download_selected_items(
        f"Preparing selected SeedVR2 model: {model_filename}",
        items,
        skip_verify=skip_verify,
        dry_run=dry_run,
    )


def ensure_flashvsr_model(
    version: str,
    *,
    vae_model: str = "Wan2.2",
    int8_convrot: bool = False,
    skip_verify: bool = False,
    dry_run: bool = False,
) -> bool:
    config = FLASHVSR_CONFIGS.get(str(version))
    if config is None:
        print(f"[ERROR] Unsupported FlashVSR version: {version}", flush=True)
        return False
    vae_spec = FLASHVSR_VAE_FILES.get(str(vae_model))
    if vae_spec is None:
        print(f"[ERROR] Unsupported FlashVSR VAE: {vae_model}", flush=True)
        return False

    repo_id = config["repo_id"]
    target_dir = config["target_dir"]
    items: List[Tuple[str, str, Path, str]] = [
        (repo_id, filename, target_dir, filename)
        for filename in ("config.json", "model_index.json", "LQ_proj_in.ckpt", "TCDecoder.ckpt")
    ]
    vae_repo, vae_remote = vae_spec
    items.append((vae_repo or repo_id, vae_remote, target_dir, Path(vae_remote).name))
    items.append((SEEDVR2_REPO_ID, "posi_prompt.pth", FLASHVSR_ROOT_DIR, "posi_prompt.pth"))

    model_dir_name = "FlashVSR-v1.1" if str(version) == "1.1" else "FlashVSR"
    if int8_convrot:
        cache_filename = PREBUILT_INT8_CONVROT_FILES[model_dir_name]
        items.append((SEEDVR2_REPO_ID, cache_filename, FLASHVSR_INT8_CACHE_DIR, cache_filename))
    else:
        weight_name = "diffusion_pytorch_model_streaming_dmd.safetensors"
        items.append((repo_id, weight_name, target_dir, weight_name))

    return _download_selected_items(
        f"Preparing selected FlashVSR v{version} model",
        items,
        skip_verify=skip_verify,
        dry_run=dry_run,
    )


def ensure_sparkvsr_model(
    model_name: str,
    *,
    skip_verify: bool = False,
    dry_run: bool = False,
) -> bool:
    normalized = str(model_name or "").strip()
    bf16_target = SPARKVSR_MODELS_DIR / SPARKVSR_S2_REPO_SUBDIR
    if normalized in {"SparkVSR-bf16", "SparkVSR-S2", "SparkVSR-fp8-scaled"}:
        items = _repo_file_items(
            SPARKVSR_S2_REPO_ID,
            bf16_target,
            include_prefix=SPARKVSR_S2_REPO_SUBDIR,
            strip_prefix=SPARKVSR_S2_REPO_SUBDIR,
        )
    elif normalized == "SparkVSR-int8-convrot":
        source_weight = f"{SPARKVSR_S2_REPO_SUBDIR}/transformer/diffusion_pytorch_model.safetensors"
        items = _repo_file_items(
            SPARKVSR_S2_REPO_ID,
            bf16_target,
            include_prefix=SPARKVSR_S2_REPO_SUBDIR,
            strip_prefix=SPARKVSR_S2_REPO_SUBDIR,
            exclude_files=[source_weight],
        )
        cache_name = PREBUILT_INT8_CONVROT_FILES["SparkVSR"]
        items.append((SEEDVR2_REPO_ID, cache_name, SPARKVSR_MODELS_DIR, cache_name))
    else:
        print(f"[ERROR] No automatic download is configured for SparkVSR model: {normalized}", flush=True)
        return False
    return _download_selected_items(
        f"Preparing selected SparkVSR model: {normalized}",
        items,
        skip_verify=skip_verify,
        dry_run=dry_run,
    )


def _ltx25_items_for_files(filenames) -> List[Tuple[str, str, Path, str]]:
    items: List[Tuple[str, str, Path, str]] = []
    for filename in filenames:
        repo_id = LTX25_FILE_SOURCES.get(filename)
        if repo_id is None:
            print(f"[ERROR] Unknown LTX 2.5 file: {filename}", flush=True)
            continue
        items.append((repo_id, filename, LTX25_MODELS_DIR, filename))
    return items


def ensure_ltx25_model(
    model_name: str,
    *,
    text_encoder: str = "Gemma 4 12B INT8 ConvRot",
    video_vae: str = "Video VAE Conv",
    skip_verify: bool = False,
    dry_run: bool = False,
) -> bool:
    """Download only the files that the selected LTX 2.5 variant needs at runtime."""
    normalized = str(model_name or "").strip()
    transformer_file = LTX25_TRANSFORMERS.get(normalized)
    if transformer_file is None and normalized in LTX25_FILE_SOURCES:
        transformer_file = normalized  # allow raw filenames too
    if transformer_file is None:
        print(f"[ERROR] No automatic download is configured for LTX 2.5 model: {normalized}", flush=True)
        print(f"        Valid options: {', '.join(sorted(LTX25_TRANSFORMERS))}", flush=True)
        return False
    te_file = LTX25_TEXT_ENCODERS.get(str(text_encoder or "").strip())
    if te_file is None:
        te_file = LTX25_TEXT_ENCODERS["Gemma 4 12B INT8 ConvRot"]
    vae_file = LTX25_VIDEO_VAES.get(str(video_vae or "").strip())
    if vae_file is None:
        vae_file = LTX25_VIDEO_VAES["Video VAE Conv"]

    wanted = [transformer_file, te_file, vae_file, *LTX25_ALWAYS_FILES]
    # Preserve order while dropping duplicates.
    wanted = list(dict.fromkeys(wanted))
    missing = [name for name in wanted if not (LTX25_MODELS_DIR / name).is_file()]
    if not missing:
        print(f"[READY] All LTX 2.5 files for '{normalized}' already exist in {LTX25_MODELS_DIR}", flush=True)
        return True
    return _download_selected_items(
        f"Preparing selected LTX 2.5 model: {normalized}",
        _ltx25_items_for_files(missing),
        skip_verify=skip_verify,
        dry_run=dry_run,
    )


def download_ltx25_all(*, skip_verify: bool = False, dry_run: bool = False) -> bool:
    """Download every LTX 2.5 file (all transformers, encoders, VAEs and extras)."""
    all_files = list(LTX25_FILE_SOURCES)
    missing = [name for name in all_files if not (LTX25_MODELS_DIR / name).is_file()]
    if not missing:
        print(f"[READY] All LTX 2.5 files already exist in {LTX25_MODELS_DIR}", flush=True)
        return True
    return _download_selected_items(
        "Preparing ALL LTX 2.5 files",
        _ltx25_items_for_files(missing),
        skip_verify=skip_verify,
        dry_run=dry_run,
    )


def ensure_gan_model(
    model_filename: str,
    *,
    skip_verify: bool = False,
    dry_run: bool = False,
) -> bool:
    model_filename = Path(str(model_filename or "")).name
    if model_filename not in BESTIMAGEUPSCALE_MODEL_FILES:
        print(f"[ERROR] No automatic download is configured for GAN model: {model_filename}", flush=True)
        return False
    return _download_selected_items(
        f"Preparing selected GAN model: {model_filename}",
        [(BESTIMAGEUPSCALE_REPO_ID, model_filename, IMAGE_UPSCALE_MODELS_DIR, model_filename)],
        skip_verify=skip_verify,
        dry_run=dry_run,
    )


def ensure_rife_model(
    version: str,
    *,
    skip_verify: bool = False,
    dry_run: bool = False,
) -> bool:
    version = str(version or "").strip()
    if version not in RIFE_VERSION_FOLDERS:
        print(f"[ERROR] No automatic download is configured for RIFE model: {version}", flush=True)
        return False
    prefix = f"{RIFE_REPO_SUBDIR}/{version}"
    items = _repo_file_items(
        RIFE_REPO_ID,
        RIFE_MODELS_DIR / version,
        include_prefix=prefix,
        strip_prefix=prefix,
    )
    return _download_selected_items(
        f"Preparing selected RIFE model: {version}",
        items,
        skip_verify=skip_verify,
        dry_run=dry_run,
    )


def get_model_choice() -> Tuple[List[str], bool, bool, bool, bool, bool]:
    """Interactive model selection.
    
    Returns:
        Tuple of (model_ids list, include_rife, include_best, include_flashvsr, include_sparkvsr, include_sparkvsr_all)
    """
    print("\n" + "=" * 60)
    print("Ultimate Video/Image Upscaler Model Downloader")
    print("=" * 60)
    print(f"\nSeedVR2 Core/FP8 Repository: {SEEDVR2_REPO_ID}")
    print(f"SeedVR2 GGUF Repository: {SEEDVR2_GGUF_REPO_ID}")
    print(f"SeedVR2 Target: {SEEDVR2_MODELS_DIR}")
    print(f"\nRIFE Repository: {RIFE_REPO_ID}/{RIFE_REPO_SUBDIR}")
    print(f"RIFE Target: {RIFE_MODELS_DIR}")
    print(f"\nBestImageUpscalers Repository: {BESTIMAGEUPSCALE_REPO_ID}")
    print(f"BestImageUpscalers Target: {IMAGE_UPSCALE_MODELS_DIR}")
    print(f"\nFlashVSR+ Repositories: {FLASHVSR_V10_REPO_ID}, {FLASHVSR_V11_REPO_ID}")
    print(f"FlashVSR+ VAE Repository: {FLASHVSR_VAE_REPO_ID}/{FLASHVSR_VAE_SUBDIR}")
    print(f"FlashVSR+ Targets: {FLASHVSR_CONFIGS['1.0']['target_dir']}, {FLASHVSR_CONFIGS['1.1']['target_dir']}")
    print(f"FlashVSR+ Auxiliary Target: {FLASHVSR_ROOT_DIR}")
    print(f"\nSparkVSR Repository: {SPARKVSR_S2_REPO_ID}/{SPARKVSR_S2_REPO_SUBDIR}")
    print(f"SparkVSR Target: {SPARKVSR_CONFIGS['s2']['target_dir']}")
    print(f"\nLTX 2.5 Repositories: {LTX25_REPO_ID}, {LTX25_INT4_REPO_ID}, {LTX25_IC_LORA_REPO_ID}")
    print(f"LTX 2.5 Target: {LTX25_MODELS_DIR}")
    print("\nAvailable models:")
    print()
    print("=" * 40)
    print("SeedVR2 Models:")
    print("=" * 40)
    print("1. VAE Model (~330 MB)")
    print("   Required for all SeedVR2 operations")
    print()
    print("2. SeedVR2 3B Model (~6.5 GB)")
    print("   Faster processing, lower VRAM requirement")
    print()
    print("3. SeedVR2 7B Model (~14 GB)")
    print("   Best quality, higher VRAM requirement")
    print()
    print("4. SeedVR2 7B Sharp Model (~14 GB)")
    print("   Sharpened variant of 7B model")
    print()
    print("5. ALL SeedVR2 CORE MODELS (~35 GB total)")
    print("   Download VAE + 3B + 7B + 7B Sharp (no FP8/GGUF)")
    print()
    print("12. SeedVR2 7B FP8 Mixed Block35 Model (~14 GB)")
    print("    FP8 e4m3fn mixed block35 variant")
    print()
    print("13. SeedVR2 7B Sharp FP8 Mixed Block35 Model (~14 GB)")
    print("    Sharpened FP8 e4m3fn mixed block35 variant")
    print()
    print("15. SeedVR2 3B GGUF Q8_0 Model (~4 GB)")
    print("    Quantized GGUF model from cmeka/SeedVR2-GGUF")
    print()
    print("16. SeedVR2 7B GGUF Q8_0 Model (~8 GB)")
    print("    Quantized GGUF model from cmeka/SeedVR2-GGUF")
    print()
    print("17. SeedVR2 7B Sharp GGUF Q8_0 Model (~8 GB)")
    print("    Sharpened quantized GGUF model from cmeka/SeedVR2-GGUF")
    print()
    print("18. ALL SEPARATE GGUF + FP8 MODELS (~48 GB)")
    print("    Downloads FP8 + GGUF bundle (separate from regular all)")
    print()
    print("6. VAE + 3B Only (~7 GB)")
    print("   Recommended for GPUs with <16GB VRAM")
    print()
    print("7. VAE + 7B Only (~14.5 GB)")
    print("   Recommended for GPUs with 16GB+ VRAM")
    print()
    print("=" * 40)
    print("RIFE Models:")
    print("=" * 40)
    print("8. RIFE Models (4.14-4.26)")
    print("   Frame interpolation models")
    print()
    print("=" * 40)
    print("Image Upscale Models:")
    print("=" * 40)
    print("10. BestImageUpscalers Models")
    print("    High-quality image upscaling models")
    print()
    print("=" * 40)
    print("FlashVSR+ Models:")
    print("=" * 40)
    print("14. FlashVSR+ Models (v1.0 + v1.1)")
    print("    Downloads both model repos + FlashVSR_VAEs and places posi_prompt.pth at the runtime root")
    print()
    print("19. SparkVSR Stage-2 BF16 Model")
    print("    Downloads MonsterMMORPG/Wan_GGUF/SparkVSR-bf16 into SparkVSR/models/SparkVSR-bf16")
    print()
    print("20. SparkVSR Stage-1 + Stage-2 BF16 Models")
    print("    Downloads Stage-1 plus MonsterMMORPG/Wan_GGUF/SparkVSR-bf16")
    print()
    print("=" * 40)
    print("LTX 2.5 Upscaler Models:")
    print("=" * 40)
    print("21. ALL LTX 2.5 Files (~200 GB total)")
    print("    Every transformer (INT8/INT4/NVFP4/BF16, dev + distilled), Gemma text")
    print("    encoders, video/audio VAEs, IC LoRA upscaler and extras into LTX25_Models.")
    print("    Tip: the app auto-downloads only what the selected LTX 2.5 variant needs;")
    print("    use --ensure-ltx25 \"Distilled INT8 ConvRot\" for a single variant (~38 GB).")
    print()
    print("=" * 40)
    print("Combined Options:")
    print("=" * 40)
    print("9. ALL REGULAR MODELS (SeedVR2 CORE + RIFE + FlashVSR+ + SparkVSR)")
    print("   Download regular set (excludes FP8/GGUF)")
    print()
    print("11. ALL REGULAR MODELS (+ BestImageUpscalers)")
    print("    Download regular set + BestImageUpscalers (excludes FP8/GGUF)")
    print()
    
    while True:
        try:
            choice = input("Please select option (1-21): ").strip()
            if choice == "21":
                if download_ltx25_all():
                    sys.exit(0)
                sys.exit(1)
            if choice == "1":
                return ["vae"], False, False, False, False, False
            elif choice == "2":
                return ["3b"], False, False, False, False, False
            elif choice == "3":
                return ["7b"], False, False, False, False, False
            elif choice == "4":
                return ["7b_sharp"], False, False, False, False, False
            elif choice == "5":
                return ALL_SEEDVR2_MODEL_IDS.copy(), False, False, False, False, False
            elif choice == "6":
                return ["vae", "3b"], False, False, False, False, False
            elif choice == "7":
                return ["vae", "7b"], False, False, False, False, False
            elif choice == "8":
                return [], True, False, False, False, False
            elif choice == "9":
                return ALL_SEEDVR2_MODEL_IDS.copy(), True, False, True, True, False
            elif choice == "10":
                return [], False, True, False, False, False
            elif choice == "11":
                return ALL_SEEDVR2_MODEL_IDS.copy(), True, True, True, True, False
            elif choice == "12":
                return ["7b_fp8_mixed_block35"], False, False, False, False, False
            elif choice == "13":
                return ["7b_sharp_fp8_mixed_block35"], False, False, False, False, False
            elif choice == "14":
                return [], False, False, True, False, False
            elif choice == "15":
                return ["3b_gguf_q8_0"], False, False, False, False, False
            elif choice == "16":
                return ["7b_gguf_q8_0"], False, False, False, False, False
            elif choice == "17":
                return ["7b_sharp_gguf_q8_0"], False, False, False, False, False
            elif choice == "18":
                return SEPARATE_GGUF_FP8_MODEL_IDS.copy(), False, False, False, False, False
            elif choice == "19":
                return [], False, False, False, True, False
            elif choice == "20":
                return [], False, False, False, True, True
            else:
                print("Invalid choice. Please enter 1-21.")
        except KeyboardInterrupt:
            print("\nDownload cancelled.")
            sys.exit(0)


def download_models(model_ids: Optional[List[str]] = None, 
                    skip_verify: bool = False,
                    dry_run: bool = False,
                    include_rife: bool = False,
                    include_best: bool = False,
                    include_flashvsr: bool = False,
                    include_sparkvsr: bool = False,
                    include_sparkvsr_all: bool = False) -> None:
    """Main download function."""
    if model_ids is None and not include_rife and not include_best and not include_flashvsr and not include_sparkvsr:
        model_ids, include_rife, include_best, include_flashvsr, include_sparkvsr, include_sparkvsr_all = get_model_choice()
    
    if model_ids is None:
        model_ids = []
    
    # Validate model IDs
    invalid_ids = [m for m in model_ids if m not in SEEDVR2_MODEL_CONFIGS]
    if invalid_ids:
        print(f"Error: Unknown model ID(s): {invalid_ids}")
        print(f"Valid options: {list(SEEDVR2_MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Download Configuration")
    print("=" * 60)
    
    if model_ids:
        seedvr2_repos = sorted(
            {
                SEEDVR2_MODEL_CONFIGS[model_id].get("repo_id", SEEDVR2_REPO_ID)
                for model_id in model_ids
            }
        )
        print("\nSeedVR2 Repositories:")
        for repo_id in seedvr2_repos:
            print(f"  - {repo_id}")
        print(f"SeedVR2 Target Directory: {SEEDVR2_MODELS_DIR}")
        print(f"SeedVR2 Models to download:")
        for model_id in model_ids:
            config = SEEDVR2_MODEL_CONFIGS[model_id]
            repo_id = config.get("repo_id", SEEDVR2_REPO_ID)
            print(f"  - {config['name']} ({config['size_hint']}) [{repo_id}]")
    
    if include_rife:
        print(f"\nRIFE Repository: {RIFE_REPO_ID}")
        print(f"RIFE Source Directory: {RIFE_REPO_SUBDIR}")
        print(f"RIFE Versions: {', '.join(RIFE_VERSION_FOLDERS)}")
        print(f"RIFE Target Directory: {RIFE_MODELS_DIR}")
        print(f"RIFE: Download selected version folders from repository")
    
    if include_best:
        print(f"\nBestImageUpscalers Repository: {BESTIMAGEUPSCALE_REPO_ID}")
        print(f"BestImageUpscalers Target Directory: {IMAGE_UPSCALE_MODELS_DIR}")
        print(f"BestImageUpscalers: Download all files from repository")

    if include_flashvsr:
        print(f"\nFlashVSR+ Repositories:")
        print(f"  - v1.0: {FLASHVSR_V10_REPO_ID} -> {FLASHVSR_CONFIGS['1.0']['target_dir']}")
        print(f"  - v1.1: {FLASHVSR_V11_REPO_ID} -> {FLASHVSR_CONFIGS['1.1']['target_dir']}")
        print(f"  - VAE folder: {FLASHVSR_VAE_REPO_ID}/{FLASHVSR_VAE_SUBDIR} -> each FlashVSR version folder")
        for aux in FLASHVSR_AUXILIARY_FILES:
            print(f"  - Auxiliary: {aux['repo_id']}/{aux['filename']} -> {aux['target_dir']}")
        print("FlashVSR+: Download all files from both model repositories + auxiliary files + shared VAE folder")

    if include_sparkvsr:
        spark_keys = ["s1", "s2"] if include_sparkvsr_all else ["s2"]
        print(f"\nSparkVSR Repositories:")
        for key in spark_keys:
            cfg = SPARKVSR_CONFIGS[key]
            source = cfg["repo_id"]
            if cfg.get("repo_subdir"):
                source = f"{source}/{cfg['repo_subdir']}"
            print(f"  - {cfg['name']}: {source} -> {cfg['target_dir']}")
        print("SparkVSR: Download Diffusers model repository files")
    
    print(f"\nSkip Verification: {skip_verify}")
    print()
    
    if dry_run:
        print("[DRY RUN] Would download the above files. Exiting.")
        return
    
    downloader = RobustDownloader(DOWNLOAD_CONFIG, skip_verify=skip_verify)
    
    total_successful = 0
    total_failed = 0
    failed_files = []
    
    # Download SeedVR2 models
    if model_ids:
        # Ensure target directory exists
        SEEDVR2_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("Downloading SeedVR2 Models")
        print("=" * 60)
        
        for i, model_id in enumerate(model_ids, 1):
            config = SEEDVR2_MODEL_CONFIGS[model_id]
            filename = config["filename"]
            repo_id = config.get("repo_id", SEEDVR2_REPO_ID)
            
            print(f"\n{'=' * 60}")
            print(f"[{i}/{len(model_ids)}] {config['name']}")
            print(f"File: {filename}")
            print(f"Repository: {repo_id}")
            print(f"Description: {config['description']}")
            print("=" * 60)
            
            if downloader.download_file(repo_id, filename, SEEDVR2_MODELS_DIR):
                total_successful += 1
            else:
                total_failed += 1
                failed_files.append(f"SeedVR2 ({repo_id}): {filename}")
    
    # Download RIFE models
    if include_rife:
        print("\n" + "=" * 60)
        print("Downloading RIFE Models (4.14-4.26)")
        print("=" * 60)
        print(f"Repository: {RIFE_REPO_ID}")
        print(f"Source Directory: {RIFE_REPO_SUBDIR}")
        print(f"Versions: {', '.join(RIFE_VERSION_FOLDERS)}")
        print(f"Target: {RIFE_MODELS_DIR}")
        print("=" * 60)
        
        rife_success, rife_fail = downloader.download_repo(
            RIFE_REPO_ID, 
            RIFE_MODELS_DIR,
            exclude_patterns=['.gitattributes', '.gitignore'],
            include_prefixes=RIFE_REPO_PREFIXES,
            strip_prefix=RIFE_REPO_SUBDIR,
        )
        total_successful += rife_success
        total_failed += rife_fail
        if rife_fail > 0:
            failed_files.append(f"RIFE: {rife_fail} file(s)")
    
    # Download BestImageUpscalers models
    if include_best:
        print("\n" + "=" * 60)
        print("Downloading BestImageUpscalers Models")
        print("=" * 60)
        print(f"Repository: {BESTIMAGEUPSCALE_REPO_ID}")
        print(f"Target: {IMAGE_UPSCALE_MODELS_DIR}")
        print("=" * 60)
        
        best_success, best_fail = downloader.download_repo(
            BESTIMAGEUPSCALE_REPO_ID,
            IMAGE_UPSCALE_MODELS_DIR,
            exclude_patterns=['.gitattributes', '.gitignore']
        )
        total_successful += best_success
        total_failed += best_fail
        if best_fail > 0:
            failed_files.append(f"BestImageUpscalers: {best_fail} file(s)")

    # Download FlashVSR+ models (v1.0 + v1.1), shared VAE folder files, and auxiliary files
    if include_flashvsr:
        print("\n" + "=" * 60)
        print("Downloading FlashVSR+ Models (v1.0 + v1.1)")
        print("=" * 60)
        for version in ["1.0", "1.1"]:
            cfg = FLASHVSR_CONFIGS[version]
            print(f"\nRepository: {cfg['repo_id']}")
            print(f"Target: {cfg['target_dir']}")
            success, fail = downloader.download_repo(
                cfg["repo_id"],
                cfg["target_dir"],
                exclude_patterns=['.gitattributes', '.gitignore']
            )
            total_successful += success
            total_failed += fail
            if fail > 0:
                failed_files.append(f"FlashVSR+ v{version}: {fail} file(s)")

            print(f"\nShared VAE folder: {FLASHVSR_VAE_SUBDIR}")
            print(f"Repository: {FLASHVSR_VAE_REPO_ID}")
            print(f"Target: {cfg['target_dir']}")
            vae_success, vae_fail = downloader.download_repo(
                FLASHVSR_VAE_REPO_ID,
                cfg["target_dir"],
                exclude_patterns=['.gitattributes', '.gitignore'],
                include_prefixes=FLASHVSR_VAE_PREFIXES,
                strip_prefix=FLASHVSR_VAE_SUBDIR,
            )
            total_successful += vae_success
            total_failed += vae_fail
            if vae_fail > 0:
                failed_files.append(f"FlashVSR+ v{version} VAE folder: {vae_fail} file(s)")

        for aux in FLASHVSR_AUXILIARY_FILES:
            print(f"\nAuxiliary file: {aux['filename']}")
            print(f"Repository: {aux['repo_id']}")
            print(f"Target: {aux['target_dir']}")
            if downloader.download_file(aux["repo_id"], aux["filename"], aux["target_dir"]):
                total_successful += 1
            else:
                total_failed += 1
                failed_files.append(f"FlashVSR+ auxiliary: {aux['filename']}")

    # Download SparkVSR models
    if include_sparkvsr:
        print("\n" + "=" * 60)
        print("Downloading SparkVSR Models")
        print("=" * 60)
        spark_keys = ["s1", "s2"] if include_sparkvsr_all else ["s2"]
        for key in spark_keys:
            cfg = SPARKVSR_CONFIGS[key]
            print(f"\nRepository: {cfg['repo_id']}")
            if cfg.get("repo_subdir"):
                print(f"Source Directory: {cfg['repo_subdir']}")
            print(f"Target: {cfg['target_dir']}")
            success, fail = downloader.download_repo(
                cfg["repo_id"],
                cfg["target_dir"],
                exclude_patterns=['.gitattributes', '.gitignore'],
                include_prefixes=cfg.get("include_prefixes"),
                strip_prefix=cfg.get("strip_prefix"),
            )
            total_successful += success
            total_failed += fail
            if fail > 0:
                failed_files.append(f"{cfg['name']}: {fail} file(s)")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Download Summary")
    print("=" * 60)
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")
    
    if total_failed == 0:
        print(f"\n✓ All downloads completed successfully!")
        
        if model_ids:
            print(f"\nSeedVR2 models saved to: {SEEDVR2_MODELS_DIR}")
            print(f"Downloaded SeedVR2 files:")
            for model_id in model_ids:
                filepath = SEEDVR2_MODELS_DIR / SEEDVR2_MODEL_CONFIGS[model_id]["filename"]
                if filepath.exists():
                    size = filepath.stat().st_size
                    print(f"  - {SEEDVR2_MODEL_CONFIGS[model_id]['filename']} ({RobustDownloader.format_bytes(size)})")
        
        if include_rife:
            print(f"\nRIFE models saved to: {RIFE_MODELS_DIR}")
        if include_best:
            print(f"\nBestImageUpscalers models saved to: {IMAGE_UPSCALE_MODELS_DIR}")
        if include_flashvsr:
            print(f"\nFlashVSR+ v1.0 models saved to: {FLASHVSR_CONFIGS['1.0']['target_dir']}")
            print(f"FlashVSR+ v1.1 models saved to: {FLASHVSR_CONFIGS['1.1']['target_dir']}")
            for aux in FLASHVSR_AUXILIARY_FILES:
                print(f"FlashVSR+ auxiliary file saved to: {aux['target_dir'] / aux['filename']}")
        if include_sparkvsr:
            print(f"\nSparkVSR models saved to: {SPARKVSR_MODELS_DIR}")
    else:
        print(f"\n⚠ Some downloads failed:")
        for f in failed_files:
            print(f"  - {f}")
        print(f"\nPlease re-run the script to retry failed downloads.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Download models for Ultimate Video/Image Upscalers Premium',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Models_Downloader.py                      # Interactive mode
  python Models_Downloader.py --all                # Download regular all (SeedVR2 core + RIFE + FlashVSR+ + SparkVSR + BestImageUpscalers)
  python Models_Downloader.py --seedvr2            # Download SeedVR2 core models only
  python Models_Downloader.py --seedvr2-gguf-fp8   # Download separate GGUF + FP8 SeedVR2 bundle
  python Models_Downloader.py --rife               # Download RIFE models (4.14-4.26) only
  python Models_Downloader.py --flashvsr           # Download FlashVSR+ model repos + FlashVSR_VAEs + posi_prompt.pth
  python Models_Downloader.py --sparkvsr           # Download SparkVSR Stage-2 BF16 model
  python Models_Downloader.py --sparkvsr-all       # Download SparkVSR Stage-1 + Stage-2 BF16 models
  python Models_Downloader.py --vae                # Download VAE model only
  python Models_Downloader.py --3b                 # Download 3B model only
  python Models_Downloader.py --7b                 # Download 7B model only
  python Models_Downloader.py --7b-sharp           # Download 7B Sharp model only
  python Models_Downloader.py --3b-gguf-q8-0       # Download 3B GGUF Q8_0 model
  python Models_Downloader.py --7b-gguf-q8-0       # Download 7B GGUF Q8_0 model
  python Models_Downloader.py --7b-sharp-gguf-q8-0 # Download 7B Sharp GGUF Q8_0 model
  python Models_Downloader.py --7b-fp8-mixed-block35
                                            # Download 7B FP8 mixed block35 model only
  python Models_Downloader.py --7b-sharp-fp8-mixed-block35
                                            # Download 7B Sharp FP8 mixed block35 model only
  python Models_Downloader.py --vae --3b           # Download VAE and 3B models
  python Models_Downloader.py --vae --rife         # Download VAE and RIFE models
  python Models_Downloader.py --vae --flashvsr     # Download VAE and FlashVSR+ models
  python Models_Downloader.py --all --xet-concurrency 32
                                            # Tune native Xet streams (default 32)
  python Models_Downloader.py --all --connections 16
                                            # Pin HTTP fallback connections (auto is 16-32)
  python Models_Downloader.py --all --skip-verify  # Download regular all, skip SHA256 verification
  python Models_Downloader.py --all --dry-run      # Show what would be downloaded
  python Models_Downloader.py --best               # Download BestImageUpscalers models
  python Models_Downloader.py --rife --best        # Download RIFE + BestImageUpscalers
  python Models_Downloader.py --ensure-seedvr2 seedvr2_ema_7b_fp16.safetensors
  python Models_Downloader.py --ensure-seedvr2 seedvr2_ema_7b_fp16.safetensors --int8-convrot
  python Models_Downloader.py --ensure-flashvsr 1.1 --flashvsr-vae Wan2.2 --int8-convrot
  python Models_Downloader.py --ensure-sparkvsr SparkVSR-int8-convrot
  python Models_Downloader.py --ensure-gan 4x-UltraSharpV2.safetensors
  python Models_Downloader.py --ensure-rife 4.26
  python Models_Downloader.py --ltx25              # Download ALL LTX 2.5 files into LTX25_Models
  python Models_Downloader.py --ensure-ltx25 "Distilled INT8 ConvRot"
  python Models_Downloader.py --ensure-ltx25 "Dev BF16" --ltx25-te "Gemma 4 12B BF16" --ltx25-vae "Video VAE Regular"
        """
    )
    parser.add_argument('--all', action='store_true',
                        help='Download regular all (SeedVR2 core + RIFE + FlashVSR+ + SparkVSR + BestImageUpscalers), excludes FP8/GGUF')
    parser.add_argument('--seedvr2', action='store_true',
                        help='Download SeedVR2 core models only (VAE + 3B + 7B + 7B Sharp), excludes FP8/GGUF')
    parser.add_argument('--seedvr2-gguf-fp8', dest='seedvr2_gguf_fp8', action='store_true',
                        help='Download separate SeedVR2 GGUF + FP8 bundle only')
    parser.add_argument('--rife', action='store_true',
                        help='Download RIFE models (4.14-4.26 folders from Wan_GGUF/RIFE_Models)')
    parser.add_argument('--flashvsr', action='store_true',
                        help='Download FlashVSR+ models (v1.0 + v1.1 repos), their VAEs, and runtime prompt weights')
    parser.add_argument('--sparkvsr', action='store_true',
                        help='Download SparkVSR Stage-2 BF16 Diffusers model from MonsterMMORPG/Wan_GGUF/SparkVSR-bf16')
    parser.add_argument('--sparkvsr-all', dest='sparkvsr_all', action='store_true',
                        help='Download SparkVSR Stage-1 and Stage-2 BF16 Diffusers models')
    parser.add_argument('--vae', action='store_true',
                        help='Download VAE model (required for SeedVR2 operations)')
    parser.add_argument('--3b', dest='model_3b', action='store_true',
                        help='Download SeedVR2 3B model')
    parser.add_argument('--7b', dest='model_7b', action='store_true',
                        help='Download SeedVR2 7B model')
    parser.add_argument('--7b-sharp', dest='model_7b_sharp', action='store_true',
                        help='Download SeedVR2 7B Sharp model')
    parser.add_argument('--7b-fp8-mixed-block35', dest='model_7b_fp8_mixed_block35', action='store_true',
                        help='Download SeedVR2 7B FP8 mixed block35 model')
    parser.add_argument('--7b-sharp-fp8-mixed-block35', dest='model_7b_sharp_fp8_mixed_block35', action='store_true',
                        help='Download SeedVR2 7B Sharp FP8 mixed block35 model')
    parser.add_argument('--3b-gguf-q8-0', dest='model_3b_gguf_q8_0', action='store_true',
                        help='Download SeedVR2 3B GGUF Q8_0 model')
    parser.add_argument('--7b-gguf-q8-0', dest='model_7b_gguf_q8_0', action='store_true',
                        help='Download SeedVR2 7B GGUF Q8_0 model')
    parser.add_argument('--7b-sharp-gguf-q8-0', dest='model_7b_sharp_gguf_q8_0', action='store_true',
                        help='Download SeedVR2 7B Sharp GGUF Q8_0 model')
    parser.add_argument('--skip-verify', action='store_true',
                        help='Skip SHA256 verification (faster but less safe)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded without downloading')
    parser.add_argument('--best', action='store_true',
                        help='Download BestImageUpscalers models (all files from repo)')
    parser.add_argument('--ensure-seedvr2', metavar='MODEL_FILE',
                        help='Download only the selected SeedVR2 model and its VAE dependency')
    parser.add_argument('--ensure-flashvsr', choices=sorted(FLASHVSR_CONFIGS),
                        help='Download only the selected FlashVSR version and runtime dependencies')
    parser.add_argument('--flashvsr-vae', choices=list(FLASHVSR_VAE_FILES), default='Wan2.2',
                        help='VAE dependency used with --ensure-flashvsr')
    parser.add_argument('--ensure-sparkvsr', metavar='MODEL_NAME',
                        help='Download only the selected SparkVSR variant and runtime dependencies')
    parser.add_argument('--ensure-gan', metavar='MODEL_FILE',
                        help='Download only the selected GAN upscaler model')
    parser.add_argument('--ensure-rife', choices=RIFE_VERSION_FOLDERS,
                        help='Download only the selected RIFE version')
    parser.add_argument('--ensure-ltx25', metavar='MODEL_NAME',
                        help='Download only the selected LTX 2.5 variant and its runtime dependencies '
                             f'({", ".join(sorted(LTX25_TRANSFORMERS))})')
    parser.add_argument('--ltx25-te', choices=list(LTX25_TEXT_ENCODERS), default='Gemma 4 12B INT8 ConvRot',
                        help='Text encoder used with --ensure-ltx25')
    parser.add_argument('--ltx25-vae', choices=list(LTX25_VIDEO_VAES), default='Video VAE Conv',
                        help='Video VAE used with --ensure-ltx25')
    parser.add_argument('--ltx25', action='store_true',
                        help='Download ALL LTX 2.5 files (transformers, encoders, VAEs, IC LoRA, extras) into LTX25_Models')
    parser.add_argument('--int8-convrot', action='store_true',
                        help='Use a prebuilt INT8 ConvRot cache for a targeted SeedVR2 or FlashVSR download')
    parser.add_argument('--connections', type=int, metavar='N', default=None,
                        help="Pin HTTP fallback byte-range connections per large file "
                             "(1-128; default automatically scales from 16 to 32)")
    parser.add_argument('--xet-concurrency', type=int, metavar='N',
                        default=DOWNLOAD_CONFIG['xet_concurrency'],
                        help=f"Native Xet download streams per large file "
                             f"(1-124, default: {DOWNLOAD_CONFIG['xet_concurrency']})")
    parser.add_argument('--no-xet', action='store_true',
                        help='Disable native Xet and use the resumable HTTP range fallback')

    args = parser.parse_args()

    if args.connections is not None and not 1 <= args.connections <= 128:
        parser.error('--connections must be between 1 and 128')
    if not 1 <= args.xet_concurrency <= 124:
        parser.error('--xet-concurrency must be between 1 and 124')
    if args.connections is not None:
        DOWNLOAD_CONFIG['num_connections'] = args.connections
        DOWNLOAD_CONFIG['http_adaptive_connections'] = 0
    DOWNLOAD_CONFIG['xet_concurrency'] = args.xet_concurrency
    if args.no_xet:
        DOWNLOAD_CONFIG['xet_enabled'] = 0

    targeted = [
        args.ensure_seedvr2,
        args.ensure_flashvsr,
        args.ensure_sparkvsr,
        args.ensure_gan,
        args.ensure_rife,
        args.ensure_ltx25,
    ]
    if sum(value is not None for value in targeted) > 1:
        parser.error('Choose only one --ensure-* model request at a time')
    if args.int8_convrot and not (args.ensure_seedvr2 or args.ensure_flashvsr):
        parser.error('--int8-convrot requires --ensure-seedvr2 or --ensure-flashvsr')

    selected_ok: Optional[bool] = None
    if args.ensure_seedvr2:
        selected_ok = ensure_seedvr2_model(
            args.ensure_seedvr2,
            int8_convrot=args.int8_convrot,
            skip_verify=args.skip_verify,
            dry_run=args.dry_run,
        )
    elif args.ensure_flashvsr:
        selected_ok = ensure_flashvsr_model(
            args.ensure_flashvsr,
            vae_model=args.flashvsr_vae,
            int8_convrot=args.int8_convrot,
            skip_verify=args.skip_verify,
            dry_run=args.dry_run,
        )
    elif args.ensure_sparkvsr:
        selected_ok = ensure_sparkvsr_model(
            args.ensure_sparkvsr,
            skip_verify=args.skip_verify,
            dry_run=args.dry_run,
        )
    elif args.ensure_gan:
        selected_ok = ensure_gan_model(
            args.ensure_gan,
            skip_verify=args.skip_verify,
            dry_run=args.dry_run,
        )
    elif args.ensure_rife:
        selected_ok = ensure_rife_model(
            args.ensure_rife,
            skip_verify=args.skip_verify,
            dry_run=args.dry_run,
        )
    elif args.ensure_ltx25:
        selected_ok = ensure_ltx25_model(
            args.ensure_ltx25,
            text_encoder=args.ltx25_te,
            video_vae=args.ltx25_vae,
            skip_verify=args.skip_verify,
            dry_run=args.dry_run,
        )
    elif args.ltx25:
        selected_ok = download_ltx25_all(
            skip_verify=args.skip_verify,
            dry_run=args.dry_run,
        )
    if selected_ok is not None:
        if not selected_ok:
            raise SystemExit(1)
        return
    
    model_ids = []
    include_rife = False
    include_best = False
    include_flashvsr = False
    include_sparkvsr = False
    include_sparkvsr_all = False
    
    if args.all:
        # Download regular set only (explicitly excludes FP8/GGUF group)
        model_ids = ALL_SEEDVR2_MODEL_IDS.copy()
        include_rife = True
        include_best = True
        include_flashvsr = True
        include_sparkvsr = True
    else:
        if args.seedvr2:
            model_ids.extend(ALL_SEEDVR2_MODEL_IDS)
        if args.seedvr2_gguf_fp8:
            model_ids.extend(SEPARATE_GGUF_FP8_MODEL_IDS)
        if args.vae:
            model_ids.append("vae")
        if args.model_3b:
            model_ids.append("3b")
        if args.model_7b:
            model_ids.append("7b")
        if args.model_7b_sharp:
            model_ids.append("7b_sharp")
        if args.model_7b_fp8_mixed_block35:
            model_ids.append("7b_fp8_mixed_block35")
        if args.model_7b_sharp_fp8_mixed_block35:
            model_ids.append("7b_sharp_fp8_mixed_block35")
        if args.model_3b_gguf_q8_0:
            model_ids.append("3b_gguf_q8_0")
        if args.model_7b_gguf_q8_0:
            model_ids.append("7b_gguf_q8_0")
        if args.model_7b_sharp_gguf_q8_0:
            model_ids.append("7b_sharp_gguf_q8_0")

        if model_ids:
            # Preserve order while removing duplicates from combined flags.
            model_ids = list(dict.fromkeys(model_ids))
        
        if args.rife:
            include_rife = True
        if args.best:
            include_best = True
        if args.flashvsr:
            include_flashvsr = True
        if args.sparkvsr:
            include_sparkvsr = True
        if args.sparkvsr_all:
            include_sparkvsr = True
            include_sparkvsr_all = True
    
    # If no arguments, run interactive mode
    if not model_ids and not include_rife and not include_best and not include_flashvsr and not include_sparkvsr:
        model_ids = None
    
    download_models(
        model_ids,
        skip_verify=args.skip_verify,
        dry_run=args.dry_run,
        include_rife=include_rife,
        include_best=include_best,
        include_flashvsr=include_flashvsr,
        include_sparkvsr=include_sparkvsr,
        include_sparkvsr_all=include_sparkvsr_all
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nDownload interrupted. Verified partial ranges were preserved; "
            "run the same command again to resume.",
            flush=True,
        )
        raise SystemExit(130)

