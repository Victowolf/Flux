import os
import time
import logging
from contextlib import asynccontextmanager

import torch
from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flux-api")

MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
DEVICE = "cpu"
DTYPE = torch.float32

# Limit CPU thread oversubscription; override with OMP_NUM_THREADS if needed.
CPU_THREADS = int(os.getenv("OMP_NUM_THREADS", "20"))
torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(max(1, min(4, CPU_THREADS)))
except RuntimeError:
    pass

pipe = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    logger.info("Loading %s on CPU with %s threads", MODEL_ID, CPU_THREADS)
    try:
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
        )
        pipe.to(DEVICE)
        pipe.set_progress_bar_config(disable=False)
        logger.info("Model loaded successfully on CPU")
    except Exception:
        logger.exception("Failed to load model")
        raise
    yield
    pipe = None


app = FastAPI(
    title="FLUX.1-schnell CPU Image API",
    description="CPU-only image generation service for FLUX.1-schnell.",
    version="1.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    height: int = Field(default=512, ge=256, le=1024)
    width: int = Field(default=512, ge=256, le=1024)
    num_inference_steps: int = Field(default=4, ge=1, le=4)
    seed: int | None = None


@app.get("/health")
def health():
    return {
        "status": "ok" if pipe is not None else "loading",
        "model": MODEL_ID,
        "device": DEVICE,
        "cpu_threads": CPU_THREADS,
    }


@app.post("/v1/images/generations")
def generate(req: GenerateRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    generator = None
    if req.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(req.seed)

    started = time.time()
    try:
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
        image = result.images[0]
        os.makedirs("/tmp/flux-output", exist_ok=True)
        filename = f"image-{int(time.time() * 1000)}.png"
        path = os.path.join("/tmp/flux-output", filename)
        image.save(path)
        elapsed = round(time.time() - started, 2)
        logger.info("Generated %s in %s seconds", filename, elapsed)
        return {
            "created": int(time.time()),
            "model": MODEL_ID,
            "device": DEVICE,
            "generation_seconds": elapsed,
            "seed": req.seed,
            "image_url": f"/outputs/{filename}",
        }
    except Exception as exc:
        logger.exception("Image generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


from fastapi.responses import FileResponse


@app.get("/outputs/{filename}")
def get_output(filename: str):
    if "/" in filename or "\\" in filename or not filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.join("/tmp/flux-output", filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png", filename=filename)
