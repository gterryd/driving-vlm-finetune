import os
import json
import random
from collections import defaultdict
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility

# ============================================
# 配置
# ============================================
DATAROOT = "data/nuscenes/v1.0-mini"          # nuScenes数据根目录
IMG_ABS_ROOT = "d:/AI_learning/multimodal-ai-learning/data/nuscenes/v1.0-mini"  # 图片绝对路径前缀
OUT_DIR = "data/sft"
MAX_DISTANCE = 40.0                            # 只保留40米内的物体
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================
# 第一部分：用devkit加载nuScenes
#
# 关键改进：不再直接读全景3D标注，而是用devkit的get_sample_data，
# 它能返回"前置摄像头实际能看到的物体"（内部做了3D→2D投影），
# 再加距离过滤，让训练答案和画面真正对齐。
# ============================================
print("加载nuScenes...")
nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)

# 把nuScenes的长类别名简化成我们关心的8类
def simplify(name):
    if name.startswith("vehicle.car"): return "car"
    if name.startswith("vehicle.truck"): return "truck"
    if name.startswith("vehicle.bus"): return "bus"
    if name.startswith("vehicle.motorcycle"): return "motorcycle"
    if name.startswith("vehicle.bicycle"): return "bicycle"
    if name.startswith("human.pedestrian"): return "pedestrian"
    if name.startswith("movable_object.trafficcone"): return "traffic cone"
    if name.startswith("movable_object.barrier"): return "barrier"
    return None

# ============================================
# 第二部分：遍历每一帧，统计前置摄像头可见物体
# ============================================
# img_to_counts = {图片绝对路径: {类别: 数量}}
img_to_counts = {}

for sample in nusc.sample:
    cam_front_token = sample["data"]["CAM_FRONT"]
    # get_sample_data返回：图片路径、视野内物体框、相机内参
    data_path, boxes, _ = nusc.get_sample_data(
        cam_front_token, box_vis_level=BoxVisibility.ANY
    )

    counts = defaultdict(int)
    for box in boxes:
        # box.center是相机坐标系下的位置，模长就是距离
        distance = np.linalg.norm(box.center)
        if distance > MAX_DISTANCE:
            continue
        cat_name = simplify(box.name)
        if cat_name:
            counts[cat_name] += 1

    # 取出图片相对路径（samples/CAM_FRONT/xxx.jpg），转成绝对路径
    filename = nusc.get("sample_data", cam_front_token)["filename"]
    abs_path = os.path.join(IMG_ABS_ROOT, filename)
    img_to_counts[abs_path] = dict(counts)

print(f"共处理 {len(img_to_counts)} 张前置摄像头图片")

# ============================================
# 第三部分：中文映射（复用）
# ============================================
cn_names = {
    "car": "轿车", "truck": "卡车", "bus": "公交车",
    "motorcycle": "摩托车", "bicycle": "自行车",
    "pedestrian": "行人", "traffic cone": "交通锥", "barrier": "路障",
}
cn_units = {
    "car": "辆", "truck": "辆", "bus": "辆",
    "motorcycle": "辆", "bicycle": "辆",
    "pedestrian": "名", "traffic cone": "个", "barrier": "个",
}

# ============================================
# 第四部分：生成回答的函数（复用）
# ============================================
def make_description(counts):
    if not counts:
        return "未检测到明显的目标物体"
    parts = [f"{num}{cn_units[cat]}{cn_names[cat]}" for cat, num in counts.items()]
    return "、".join(parts)


def make_risk(counts):
    risks = []
    if counts.get("pedestrian", 0) > 0:
        risks.append("场景中存在行人，需注意避让")
    if counts.get("car", 0) + counts.get("truck", 0) + counts.get("bus", 0) >= 3:
        risks.append("前方车辆较多，建议保持安全车距")
    if counts.get("traffic cone", 0) > 0:
        risks.append("存在交通锥，可能有施工或事故区域")
    if counts.get("barrier", 0) > 0:
        risks.append("存在路障，注意绕行")
    if counts.get("motorcycle", 0) > 0 or counts.get("bicycle", 0) > 0:
        risks.append("存在两轮车辆，注意盲区")
    if not risks:
        risks.append("当前场景风险较低，保持正常驾驶即可")
    return "。".join(risks) + "。"

# ============================================
# 第五部分：生成训练数据（sharegpt格式）
# ============================================
random.seed(42)
sft_data = []

for img_path, counts in img_to_counts.items():
    desc = make_description(counts)
    risk = make_risk(counts)

    # 题目1：场景描述
    sft_data.append({
        "messages": [
            {"role": "user", "content": "<image>请描述这个自动驾驶场景中的目标物体。"},
            {"role": "assistant", "content": f"该场景中包含{desc}。"}
        ],
        "images": [img_path]
    })

    # 题目2：风险分析
    sft_data.append({
        "messages": [
            {"role": "user", "content": "<image>分析这个驾驶场景的潜在风险。"},
            {"role": "assistant", "content": risk}
        ],
        "images": [img_path]
    })

    # 题目3：计数问题（只在有物体时生成）
    if counts:
        cat = random.choice(list(counts.keys()))
        num = counts[cat]
        sft_data.append({
            "messages": [
                {"role": "user", "content": f"<image>这个场景中有多少{cn_units[cat]}{cn_names[cat]}？"},
                {"role": "assistant", "content": f"{num}{cn_units[cat]}。"}
            ],
            "images": [img_path]
        })

print(f"共生成 {len(sft_data)} 条训练数据")

# ============================================
# 第六部分：保存
# ============================================
out_path = os.path.join(OUT_DIR, "driving_sft.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(sft_data, f, ensure_ascii=False, indent=2)

print(f"已保存到 {out_path}")
print("示例数据：")
print(json.dumps(sft_data[0], ensure_ascii=False, indent=2))
