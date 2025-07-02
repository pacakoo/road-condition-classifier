import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipModel
import torch.nn.functional as F
from tqdm import tqdm

# 设置设备与模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Salesforce/blip-itm-base-coco"
model = BlipModel.from_pretrained(model_name).to(device)
processor = BlipProcessor.from_pretrained(model_name)

# Step 1: 加载图像集并提取特征（仅做一次）
image_folder = r"D:\RSRD-dense\train\2023-04-08-04-46-16\left"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]

image_features = []
image_names = []

for path in tqdm(image_paths[:100], desc="提取图像特征"):  # 限制前100张，防止爆显存
    try:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
            feat = F.normalize(feat, dim=-1)  # 归一化特征
        image_features.append(feat)
        image_names.append(os.path.basename(path))
    except Exception as e:
        print(f"❌ {path}: {e}")

image_features = torch.cat(image_features, dim=0)  # shape: [num_images, feature_dim]

# Step 2: 输入一句文本 ➜ 检索匹配图像
query = "Potholed road"  # ← 你可以改成中文如“坑洼路面”
text_inputs = processor(text=query, return_tensors="pt").to(device)
with torch.no_grad():
    text_feature = model.get_text_features(**text_inputs)
    text_feature = F.normalize(text_feature, dim=-1)

# Step 3: 计算相似度
similarity = torch.matmul(image_features, text_feature.T).squeeze()  # shape: [num_images]

# Step 4: 获取Top-K结果
top_k = 5
top_indices = similarity.topk(top_k).indices.tolist()

print("\n🔍 查询结果（Top-K图像）:")
for i in top_indices:
    print(f"{image_names[i]}  -  相似度: {similarity[i].item():.4f}")
