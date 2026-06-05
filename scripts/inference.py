import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

print("加载模型中...")

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
)
model = PeftModel.from_pretrained(base_model, "saves/qwen2.5-vl-3b/lora/driving")
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    use_fast=True,
    min_pixels=256*28*28,
    max_pixels=512*28*28,
)
model.eval()
print("模型加载完成！")

img_path = "d:/AI_learning/multimodal-ai-learning/data/nuscenes/v1.0-mini/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"

questions = [
    "请描述这个自动驾驶场景中的目标物体。",
    "分析这个驾驶场景的潜在风险。",
]

for question in questions:
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": question},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    output_ids_trimmed = output_ids[0][inputs.input_ids.shape[1]:]
    answer = processor.decode(output_ids_trimmed, skip_special_tokens=True)

    print(f"\n问题：{question}")
    print(f"回答：{answer}")
