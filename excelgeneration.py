import os
import pandas as pd

# 设置你的根目录
root_folder = r"D:\RSRD-dense\train"
image_paths = []

# 三层目录遍历：train -> 每个时间戳文件夹 -> left -> *.jpg
for session_folder in os.listdir(root_folder):
    session_path = os.path.join(root_folder, session_folder)
    if os.path.isdir(session_path):
        left_folder = os.path.join(session_path, "left")  # 固定是 left 文件夹
        if os.path.isdir(left_folder):
            for filename in os.listdir(left_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(left_folder, filename)
                    image_paths.append(image_path)

# 创建 DataFrame
df = pd.DataFrame({
    "image_path": image_paths,
    "label": [""] * len(image_paths)  # 空 label，手动标注用
})

# 保存 Excel
output_excel = r"image_labels.xlsx"
df.to_excel(output_excel, index=False)

print(f"✅ 已成功写入 {len(image_paths)} 张图像路径到 Excel 文件：{output_excel}")
