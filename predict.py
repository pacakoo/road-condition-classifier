import os
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ====== 配置路径 ======
model_path = "resnet18_road_classifier.pth"     # 模型路径
excel_path = "image_labels.xlsx"                # 原始 Excel，包含 label 中文标签
test_folder = r"D:\RSRD-dense\test\left"        # 待推理的图像文件夹
output_excel = "batch_predictions.xlsx"         # 结果保存路径

# ====== 图像预处理（必须和训练一致）======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ====== 获取标签索引映射 ======
df = pd.read_excel(excel_path)
labels = sorted(df['label'].dropna().unique())
index_to_label = {idx: label for idx, label in enumerate(labels)}

# ====== 加载模型 ======
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ====== 遍历测试图像并推理 ======
results = []

for filename in tqdm(os.listdir(test_folder)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(test_folder, filename)

        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                predicted_index = outputs.argmax(dim=1).item()
                predicted_label = index_to_label[predicted_index]

            results.append({
                "image": filename,
                "path": image_path,
                "prediction": predicted_label
            })
        except Exception as e:
            print(f"[跳过] 无法处理 {filename}: {e}")

# ====== 保存结果到 Excel ======
result_df = pd.DataFrame(results)
result_df.to_excel(output_excel, index=False)

print(f"✅ 已完成批量推理，结果保存在：{output_excel}")
