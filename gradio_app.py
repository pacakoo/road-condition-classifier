import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image

# 加载训练好的模型
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, 9)  # 修改为你的类别数（比如这里是10类）
model.load_state_dict(torch.load(r"D:\研\FISSCO\AIModel\BLIP\mydemo\resnet18_road_classifier.pth", map_location='cpu'))
model.eval()

# 类别名称（按你的标注顺序写）
class_names = ['平整', '坑洼', '碎石', '杂物', '裂缝', '减速带', '井盖', '路障', '路沿', '阴影但平整']

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 推理函数
def predict(image):
    image = transform(image).unsqueeze(0)  # 添加 batch 维度
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
    return label

# Gradio 界面
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs="text",
                    title="路面图像识别系统",
                    description="上传一张图像，模型将判断路面类型。")

if __name__ == "__main__":
    demo.launch()
