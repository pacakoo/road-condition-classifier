
# 🛣️ Road Condition Classifier

使用 ResNet18 模型对车辆行驶过程中拍摄到的道路图像进行分类。支持识别包括：

> 平整、坑洼、碎石、杂物、裂缝、减速带、井盖、路障、路沿、阴影但平整

## 🚀 项目亮点

- ✅ 使用 ResNet18 模型迁移学习，轻量高效
- 📸 自主采集、标注并构建图像数据集
- 🧪 支持批量推理和图像上传预测
- 💡 Gradio 可视化界面，便于展示和交互
- 📦 项目结构清晰，利于简历附带/面试演示

---

## 📁 项目结构

```
BLIP/
├── mydemo/
│   ├── train.py                  # 模型训练脚本
│   ├── predict_batch.py          # 批量推理脚本
│   ├── gradio_app.py             # Gradio 可视化推理
│   ├── resnet18_road_classifier.pth  # 训练好的模型
│   └── image_labels.xlsx         # 图像路径与人工标注
```

---

## 🔧 环境安装

建议使用 Python 3.8+，并提前安装好 PyTorch。

```bash
pip install -r requirements.txt
```

---

## 🧪 模型训练

```bash
python train.py
```

确保你已经准备好 `image_labels.xlsx`，其中包括部分人工标注的图像路径和标签。

---

## 🧠 批量图像推理

```bash
python predict_batch.py
```

该脚本将读取文件夹中的图片，批量输出每张图像的预测类别。

---

## 🎨 启动 Gradio 可视化界面

```bash
python gradio_app.py
```

会弹出一个网页，允许你上传任意图像，查看预测结果。

---

## 🧭 未来方向

- [ ] 用更多数据 Fine-tune 模型，提高精度
- [ ] 引入 YOLOv5 进行目标检测级别的路面定位
- [ ] 尝试 BLIP 等多模态模型进行图文检索扩展
- [ ] 加入数据增强，提高模型鲁棒性

---

## 👩‍💻 亮点

- “我们用 ResNet18 作为 backbone 做轻量化分类器，迁移自 ImageNet，训练自己标注的路面图像。”
- “为提升交互体验，我们构建了 Gradio 界面，适合现场演示和部署展示。”
- “这是一个完整闭环的小项目，从数据标注、模型训练、推理到前端交互。”

---

## 📮 联系方式

- GitHub: [pacakoo](https://github.com/pacakoo)
- Email: <13812965772@163.com>
