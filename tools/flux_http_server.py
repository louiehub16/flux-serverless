# flux_http_server.py — SaladCloud-compatible FLUX server
# Serves a plain HTTP API on port 8000 (Salad container groups need HTTP, not the RunPod handler protocol).
import base64, io, os, torch
from PIL import Image
from diffusers import FluxPipeline
import threading

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")

print("Loading FLUX pipeline...", flush=True)
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
if torch.cuda.is_available():
    pipe.to("cuda")
print("FLUX ready.", flush=True)

exec_lock = threading.Lock()

class GenRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance_scale: float = 3.5
    seed: int = 42
    image: str = ""  # optional base64 png for img2img

app = FastAPI(title="FLUX Server")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenRequest):
    with exec_lock:
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
            init = Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB")
            gen_kwargs["image"] = init
        with torch.inference_mode():
            result = pipe(**gen_kwargs)
        img = result.images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode(), "format": "png",
                "width": img.width, "height": img.height}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    # IMPORTANT: bind to dual-stack IPv6 "::" so Salad's container gateway (which routes
    # inbound over IPv6) can reach the server. Binding "0.0.0.0" (IPv4 only) makes the
    # gateway's IPv6 health probes time out and the public URL never gets exposed.
    uvicorn.run(app, host="::", port=port)
