# Road Surface Classification with ResNet18

🚗 本项目基于 ResNet18 构建路面类型图像分类模型，用于自动识别车辆行驶路径中的典型路面类型，例如：坑洼、碎石、减速带、裂缝等。支持 Gradio 图形界面展示，适合用于智能驾驶、悬架控制等场景。

---

## 🧠 项目简介

- 任务：根据图像识别路面类型
- 模型：ResNet18，使用 ImageNet 预训练
- 类别：平整、坑洼、碎石、杂物、裂缝、减速带、井盖、路障、路沿、阴影
- 工具：PyTorch + torchvision + Gradio
- 数据标注：使用 Excel 格式标注图像路径和标签

---

## 📦 目录结构

