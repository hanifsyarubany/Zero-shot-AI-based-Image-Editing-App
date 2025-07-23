from setup import *
from prompt_template import *

def llama4_inference(messages, token=1024):
    completion = client_groq.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=messages,
        temperature=0.1,
        max_completion_tokens=token,
        top_p=1,
        stream=True,
        stop=None,
    )
    inference_result = ""
    for chunk in completion:
        chunk_inference = chunk.choices[0].delta.content or ""
        inference_result += chunk_inference
    text = inference_result
    return text

def base64_conversion(pil_img):
    buffer = BytesIO()
    # If format is unknown, use PNG to avoid JPEG compression artifacts
    format = pil_img.format or "PNG"
    # Optional: force convert to RGB to avoid issues with transparency in JPEG
    if format.upper() == "JPEG" and pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffer, format=format, quality=95)  # high quality for JPEG
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_base64}"

def raw_base64_conversion(pil_img):
    buffer = BytesIO()
    # If format is unknown, use PNG to avoid JPEG compression artifacts
    format = pil_img.format or "PNG"
    # Optional: force convert to RGB to avoid issues with transparency in JPEG
    if format.upper() == "JPEG" and pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffer, format=format, quality=95)  # high quality for JPEG
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64

def pil_converstion(base64_img):
    # Decode the base64 string
    image_data = base64.b64decode(base64_img)
    # Wrap the binary data with BytesIO
    image_io = BytesIO(image_data)
    # Open the image using PIL
    image = Image.open(image_io)
    # Output
    return image

def generate_image_mask(user_query, query_image):
    """ 1. Object Localization """
    # Construct messages
    messages = [
        {
            "role": "system",
            "content": system_instruction_segmentation
        },
        {
            "role": "user",
            "content": [{
                        "type": "text",
                        "text": f"User: {user_query}"
                         },
                        {
                        "type": "image_url",
                        "image_url": {"url": base64_conversion(query_image)}
                        }]  
        }
    ]
    # Creating segmentation prompt
    segmentation_prompt = llama4_inference(messages)
    # Prepare image (Image Resizing)
    original_width, original_height = query_image.size
    x_factor, y_factor = original_width/resize_size, original_height/resize_size
    resized_query_image = query_image.resize((resize_size, resize_size), Image.BILINEAR)
    # Format text based on template
    formatted_text = DETECTION_TEMPLATE.format(
        Instruction=segmentation_prompt,
        Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
    )
    # Construct messages
    message = [{
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": resized_query_image
                    },
                    {   
                        "type": "text",
                        "text": formatted_text
                    }
                ]
            }]
    # VisionReasoner inference
    inputs = vision_reasoner_processor(
            text=vision_reasoner_processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True),
            images=resized_query_image,
            padding=True,
            return_tensors="pt",
        ).to(device)
    generated_ids = vision_reasoner.generate(**inputs, use_cache=True, max_new_tokens=2048, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = vision_reasoner_processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False)[0]
    # Data Extraction
    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    pred_bboxes = []
    pred_points = []
    pred_answer = None
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            pred_answer = data
            pred_bboxes = [[
                int(item['bbox_2d'][0] * x_factor + 0.5),
                int(item['bbox_2d'][1] * y_factor + 0.5),
                int(item['bbox_2d'][2] * x_factor + 0.5),
                int(item['bbox_2d'][3] * y_factor + 0.5)
            ] for item in data]
            pred_points = [[
                int(item['point_2d'][0] * x_factor + 0.5),
                int(item['point_2d'][1] * y_factor + 0.5)
            ] for item in data]
        except Exception as e:
            print(f"Error parsing JSON: {e}")
    bboxes, points = pred_bboxes, pred_points

    """ 2. Image Segmentation """
    segmentation_model.set_image(query_image)
    img_height, img_width = query_image.height, query_image.width
    mask_arr = np.zeros((img_height, img_width), dtype=bool)
    for bbox, point in zip(bboxes, points):
        masks, scores, _ = segmentation_model.predict(
            point_coords=[point],
            point_labels=[1],
            box=bbox)
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        mask = masks[0].astype(bool)
        mask_arr = np.logical_or(mask_arr, mask)
    mask_image = Image.fromarray(mask_arr).convert('L')

    # Return Output
    return mask_image
    