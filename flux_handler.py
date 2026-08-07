import runpod
import base64, io, os, torch
from PIL import Image
from diffusers import FluxPipeline

MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")

print("Loading FLUX pipeline...")
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
if torch.cuda.is_available():
    pipe.to("cuda")
print("FLUX ready.")

def handler(job):
    inp = job["input"]
    prompt = inp.get("prompt", "")
    width = int(inp.get("width", 1024))
    height = int(inp.get("height", 1024))
    steps = int(inp.get("steps", 28))
    guidance = float(inp.get("guidance_scale", 3.5))
    seed = int(inp.get("seed", 42))
    base64_img = inp.get("image")

    gen_kwargs = dict(
        prompt=prompt,
        width=width, height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=1,
    )
    if seed >= 0:
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        gen_kwargs["generator"] = generator

    # If a base image is provided, use img2img (init_image)
    if base64_img:
        init = Image.open(io.BytesIO(base64.b64decode(base64_img))).convert("RGB")
        gen_kwargs["image"] = init

    with torch.inference_mode():
        result = pipe(**gen_kwargs)
    img = result.images[0]

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_out = base64.b64encode(buf.getvalue()).decode()
    return {"image": b64_out, "format": "png", "width": img.width, "height": img.height}

runpod.serverless.start({"handler": handler})
