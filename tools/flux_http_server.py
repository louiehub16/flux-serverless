# flux_http_server.py — SaladCloud-compatible FLUX server (CRASH-PROOF v2)
# Key fix vs exit-1 crash: start the HTTP server IMMEDIATELY (uvicorn in a thread) so
# /health responds and the process never exits, while FLUX loads in the BACKGROUND.
# Salad's readiness probe passes right away (server is up), and /generate returns 503
# until the model is ready. This stops the "Instance Exited:1" crash-loop.
import base64, io, os, threading, time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI(title="FLUX Server")
MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")

# --- model state (loaded in background) ---
_pipe = None
_model_error = None
_loading = False

class GenRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance_scale: float = 3.5
    seed: int = 42
    image: str = ""  # optional base64 PNG for img2img

def _load_model():
    global _pipe, _model_error, _loading
    _loading = True
    try:
        import torch
        from diffusers import FluxPipeline
        print("Loading FLUX pipeline...", flush=True)
        _pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        if torch.cuda.is_available():
            _pipe.to("cuda")
        print("FLUX READY.", flush=True)
    except Exception as e:
        import traceback
        _model_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print("MODEL FAILED:", _model_error, flush=True)
    finally:
        _loading = False

@app.get("/health")
def health():
    # 200 immediately (server up). This is what Salad's readiness probe hits.
    return {"status": "ok", "model_ready": _pipe is not None, "loading": _loading}

@app.post("/generate")
def generate(req: GenRequest):
    if _pipe is None:
        code = 503
        detail = "Model still loading"
        if _model_error:
            detail = f"Model failed to load: {_model_error[:300]}"
            code = 500
        return JSONResponse(status_code=code, content={"error": detail})
    import torch
    import threading as _t
    with _t.Lock():
        gen_kwargs = dict(
            prompt=req.prompt,
            width=req.width, height=req.height,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            num_images_per_prompt=1,
        )
        if req.seed >= 0:
            gen_kwargs["generator"] = torch.Generator(
                device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(req.seed)
        if req.image:
            init = __import__("PIL").Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB")
            gen_kwargs["image"] = init
        with torch.inference_mode():
            result = _pipe(**gen_kwargs)
        img = result.images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode(), "format": "png",
                "width": img.width, "height": img.height}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    # Start the HTTP server FIRST in a thread (so /health is up immediately and the
    # process never exits), load FLUX in the background.
    threading.Thread(target=_load_model, daemon=True).start()
    # bind to dual-stack IPv6 "::" for Salad's IPv6 gateway
    uvicorn.run(app, host="::", port=port, log_level="info")
