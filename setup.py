from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from diffusers import AutoPipelineForInpainting
from ip_adapter import IPAdapterXL
from ultralytics import YOLOWorld
from openai import OpenAI
from io import BytesIO
from PIL import Image
from groq import Groq
import numpy as np
import base64
import torch
import re
import json
import os

# Define the Device
resize_size = 840
device = "cuda" if torch.cuda.is_available() else "cpu"

# Vision Reasoner 7B
vision_reasoner_path="Ricky06662/VisionReasoner-7B"
vision_reasoner = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        vision_reasoner_path,
                        torch_dtype=torch.float16,
                        device_map=device).eval()
vision_reasoner_processor = AutoProcessor.from_pretrained(
                        vision_reasoner_path, 
                        padding_side="left")
# SAM-2 by Meta
segmentation_model_path ="facebook/sam2-hiera-large"
segmentation_model = SAM2ImagePredictor.from_pretrained(
                        segmentation_model_path)
# SDXL
diffusion_model_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
diffusion_model = AutoPipelineForInpainting.from_pretrained(
                        diffusion_model_path, 
                        torch_dtype=torch.float16,
                        variant="fp16").to(device)
# PrefPaint
prefpaint_path = 'kd5678/prefpaint-v1.0'
prefpaint_pipe = AutoPipelineForInpainting.from_pretrained(
                        prefpaint_path).to(device)
# IP Adapter
ip_checkpoint_path = "IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
image_encoder_path = "IP-Adapter/sdxl_models/image_encoder"
ip_model = IPAdapterXL(
                        diffusion_model, 
                        image_encoder_path, 
                        ip_checkpoint_path, 
                        device)
# Llama4 Client
os.environ["GROQ_API_KEY"] = "<your-api-key>"
client_groq = Groq()