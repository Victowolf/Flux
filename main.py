
import os
import time
import base64
import logging
import threading
import resource
from io import BytesIO
from contextlib import asynccontextmanager
from typing import Optional

import psutil
import torch
from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ==================================================
# CONFIGURATION
# ==================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("flux-api")

MODEL_ID = os.getenv(
    "MODEL_ID",
    "black-forest-labs/FLUX.1-schnell",
)

DEVICE = "cpu"

DTYPE_NAME = os.getenv("TORCH_DTYPE", "bfloat16").lower()

DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

if DTYPE_NAME not in DTYPE_MAP:
    raise ValueError("TORCH_DTYPE must be float32 or bfloat16")

DTYPE = DTYPE_MAP[DTYPE_NAME]

CPU_THREADS = max(1, int(os.getenv("OMP_NUM_THREADS", "20")))

MEMORY_INTERVAL = max(
    1,
    int(os.getenv("MEMORY_LOG_INTERVAL_SECONDS", "2")),
)

MAX_PROMPT_CHARS = 2000

torch.set_num_threads(CPU_THREADS)

try:
    torch.set_num_interop_threads(max(1, min(4, CPU_THREADS)))
except RuntimeError:
    pass

pipe: Optional[FluxPipeline] = None

generation_lock = threading.Lock()
monitor_stop = threading.Event()
monitor_thread = None

process = psutil.Process(os.getpid())


# ==================================================
# MEMORY MONITOR
# ==================================================

def to_gib(value):
    return round(float(value) / (1024 ** 3), 2)


def read_memory_value(path):
    try:
        value = open(path).read().strip()

        if value == "max":
            return None

        return int(value)

    except (OSError, ValueError):
        return None


def get_container_memory():
    # Kubernetes commonly uses cgroup v2.
    usage = read_memory_value(
        "/sys/fs/cgroup/memory.current"
    )

    limit = read_memory_value(
        "/sys/fs/cgroup/memory.max"
    )

    if usage is not None:
        return usage, limit

    # Fallback for cgroup v1.
    usage = read_memory_value(
        "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    )

    limit = read_memory_value(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    )

    if limit is not None and limit >= (1 << 60):
        limit = None

    return usage, limit


def log_memory(label="periodic"):
    try:
        memory = process.memory_info()
        system = psutil.virtual_memory()

        try:
            peak_raw = resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss

            # Linux reports ru_maxrss in KiB.
            if os.uname().sysname == "Darwin":
                peak_bytes = peak_raw
            else:
                peak_bytes = peak_raw * 1024

            peak_rss = f"{to_gib(peak_bytes)} GiB"

        except Exception:
            peak_rss = "unavailable"

        container_usage, container_limit = get_container_memory()

        if container_usage is None:
            container_memory = "unavailable"
        else:
            container_memory = f"{to_gib(container_usage)} GiB"

            if container_limit is not None:
                container_memory += (
                    f"/{to_gib(container_limit)} GiB"
                )

        logger.info(
            "[MEMORY] %s | RSS=%.2f GiB | VMS=%.2f GiB | "
            "Peak RSS=%s | Container=%s | "
            "Host available=%.2f GiB | Host used=%.1f%%",
            label,
            to_gib(memory.rss),
            to_gib(memory.vms),
            peak_rss,
            container_memory,
            to_gib(system.available),
            system.percent,
        )

    except Exception:
        logger.exception("Failed to collect memory statistics")


def memory_monitor():
    log_memory("monitor started")

    while not monitor_stop.wait(MEMORY_INTERVAL):
        log_memory("periodic")


def start_memory_monitor():
    global monitor_thread

    monitor_stop.clear()

    monitor_thread = threading.Thread(
        target=memory_monitor,
        name="memory-monitor",
        daemon=True,
    )

    monitor_thread.start()


def stop_memory_monitor():
    monitor_stop.set()

    if monitor_thread and monitor_thread.is_alive():
        monitor_thread.join(timeout=3)


# ==================================================
# MODEL LIFECYCLE
# ==================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe

    logger.info(
        "Loading FLUX model | Model=%s | Device=%s | "
        "Dtype=%s | CPU threads=%d",
        MODEL_ID,
        DEVICE,
        DTYPE_NAME,
        CPU_THREADS,
    )

    start_memory_monitor()
    log_memory("before model loading")

    try:
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )

        log_memory("after model loading")

        pipe.to(DEVICE)
        pipe.set_progress_bar_config(disable=False)

        log_memory("model ready")

        logger.info("FLUX model loaded successfully")

        yield

    except Exception:
        logger.exception("Model loading or application failed")
        raise

    finally:
        pipe = None

        stop_memory_monitor()

        log_memory("application shutdown")


app = FastAPI(
    title="FLUX.1-schnell Image Generation API",
    description=(
        "CPU-based FLUX image generation API. "
        "Returns Base64-encoded PNG data without saving images."
    ),
    version="2.1.0",
    lifespan=lifespan,
)


# ==================================================
# REQUEST SCHEMA
# ==================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=MAX_PROMPT_CHARS,
    )

    height: int = Field(
        default=512,
        ge=256,
        le=1024,
    )

    width: int = Field(
        default=512,
        ge=256,
        le=1024,
    )

    num_inference_steps: int = Field(
        default=4,
        ge=1,
        le=4,
    )

    seed: int | None = None


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health():
    return {
        "status": "ok" if pipe is not None else "loading",
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": DTYPE_NAME,
        "cpu_threads": CPU_THREADS,
        "memory_log_interval_seconds": MEMORY_INTERVAL,
    }


# ==================================================
# IMAGE GENERATION
# ==================================================

@app.post("/v1/images/generations")
def generate(req: GenerateRequest):
    if pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading",
        )

    # Prevent concurrent requests from increasing peak RAM.
    if not generation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Generation already in progress. Retry shortly.",
        )

    started = time.time()

    try:
        generator = None

        if req.seed is not None:
            generator = torch.Generator(
                device="cpu"
            ).manual_seed(req.seed)

        logger.info(
            "Starting generation | Size=%dx%d | Steps=%d | "
            "Seed=%s | Prompt length=%d",
            req.width,
            req.height,
            req.num_inference_steps,
            req.seed,
            len(req.prompt),
        )

        log_memory("before inference")

        with torch.inference_mode():
            result = pipe(
                prompt=req.prompt,
                height=req.height,
                width=req.width,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=0.0,
                max_sequence_length=256,
                generator=generator,
            )

        log_memory("after inference")

        # Encode PNG in memory. No image files are created.
        image = result.images[0]

        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()

        image_base64 = base64.b64encode(
            png_bytes
        ).decode("ascii")

        image_size_bytes = len(png_bytes)

        # Release references to image objects and PNG bytes.
        del result
        del image
        del png_bytes

        elapsed = round(time.time() - started, 2)

        log_memory("after Base64 encoding")

        logger.info(
            "Generation completed in %.2f seconds | "
            "PNG size=%.2f MiB | Base64 length=%d",
            elapsed,
            image_size_bytes / (1024 ** 2),
            len(image_base64),
        )

        return {
            "created": int(time.time()),
            "model": MODEL_ID,
            "device": DEVICE,
            "generation_seconds": elapsed,
            "seed": req.seed,
            "mime_type": "image/png",
            "image_size_bytes": image_size_bytes,
            "image_base64": image_base64,
        }

    except Exception as exc:
        log_memory("after generation exception")

        logger.exception("Image generation failed")

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    finally:
        generation_lock.release()
