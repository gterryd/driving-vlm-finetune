import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import gradio as gr

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = "saves/qwen2.5-vl-3b/lora/driving"

print("加载模型中...")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cuda",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    use_fast=True,
    min_pixels=256*28*28,
    max_pixels=512*28*28,
)
model.eval()
print("模型加载完成！")


def analyze(image, question):
    if image is None:
        return "请先上传一张驾驶场景图片。"
    if not question or not question.strip():
        return "请输入问题。"

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    trimmed = output_ids[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True)


with gr.Blocks(title="自动驾驶场景理解") as demo:
    gr.Markdown("# 自动驾驶场景理解系统")
    gr.Markdown("基于 Qwen2.5-VL-3B + LoRA 微调，分析驾驶场景中的目标物体和潜在风险。")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="上传驾驶场景图片")
            question_input = gr.Textbox(label="问题", value="请描述这个自动驾驶场景中的目标物体。")
            gr.Examples(
                examples=[
                    ["请描述这个自动驾驶场景中的目标物体。"],
                    ["分析这个驾驶场景的潜在风险。"],
                    ["这个场景中有多少辆轿车？"],
                ],
                inputs=question_input,
            )
            submit_btn = gr.Button("分析", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="分析结果", lines=10)
    submit_btn.click(fn=analyze, inputs=[image_input, question_input], outputs=output)


if __name__ == "__main__":
    demo.launch()
