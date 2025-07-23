from setup import *
from function import *
from prompt_template import *
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

app = FastAPI()

class DiffusionRequest(BaseModel):
    user_query: str
    base64_img: str

class ReferenceDiffusionRequest(BaseModel):
    user_query: str
    base64_img_query: str
    base64_img_ref: str

@app.post("/diffusion-prompt")
def prompt_based_diffusion_inference(req: DiffusionRequest):
    # Convert to PIL
    query_image = pil_converstion(req.base64_img)
    # Generate image mask
    mask_image = generate_image_mask(req.user_query, query_image)

    """ 1. Diffusion Prompt Generation """
    # Construct messages template
    messages = [
        {
            "role": "system",
            "content": system_instruction_diffusion
        },
        {
            "role": "user",
            "content": [{
                        "type": "image_url",
                        "image_url": {"url": base64_conversion(query_image)}
                        },
                        {
                        "type": "image_url",
                        "image_url": {"url": base64_conversion(mask_image)}
                        },
                        {
                        "type": "text",
                        "text": f"User: {req.user_query}"
                         }]  
        }
    ]
    # Llama4 inference
    diffusion_prompt = llama4_inference(messages)
    # Extract informaton
    diffusion_prompt_payload = json.loads(re.search(r'{.*}', diffusion_prompt, re.DOTALL).group(0))

    """ 2. Diffusion Model Inference """
    # Diffusion pipeline
    result_image = prefpaint_pipe(
      prompt=diffusion_prompt_payload["positive_prompt"],
      negative_prompt= diffusion_prompt_payload["negative_prompt"],
      image=query_image,
      mask_image=mask_image,
      eta=1.0,
      padding_mask_crop=5,
      generator=torch.Generator(device=device).manual_seed(0)
    ).images[0]
    # Convert to base64 and return the output
    return {"base64_img":raw_base64_conversion(result_image)}

@app.post("/diffusion-reference")
def reference_based_diffusion_inference(req: ReferenceDiffusionRequest):
    # Convert to PIL
    query_image = pil_converstion(req.base64_img_query)
    ref_image = pil_converstion(req.base64_img_ref)
    # Generate image mask
    mask_image = generate_image_mask(req.user_query, query_image)
    # Diffusion model inference
    result_image = ip_model.generate(
        pil_image=ref_image,
        num_samples=1,
        num_inference_steps=50,
        seed=0,
        image=query_image,
        mask_image=mask_image,
        padding_mask_crop=5,
        guidance_scale=6.5,
        strength=0.99)[0]    
    # Convert to base64 and return the output
    return {"base64_img":raw_base64_conversion(result_image)}

# Optional: health check
@app.get("/")
def root():
    return {"message": "Diffusion API is running 🚀"}