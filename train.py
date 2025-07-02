import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# --------------------------- 路面数据集定义 --------------------------- #
class RoadDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labels = sorted(self.dataframe['label'].dropna().unique())
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.encoded_labels = [self.label_to_index[label] for label in self.dataframe['label']]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.loc[idx, 'image_path']
        image = Image.open(image_path).convert("RGB")
        label = self.encoded_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --------------------------- 训练流程入口 --------------------------- #
if __name__ == '__main__':
    # ---------- 参数配置 ----------
    excel_path = "image_labels.xlsx"  # 你的Excel标签文件路径
    batch_size = 16
    num_epochs = 10
    lr = 1e-4
    num_workers = 0  # Win系统推荐设为0，Linux可设为4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ---------- 加载标注数据 ----------
    df = pd.read_excel(excel_path)
    df = df[df['label'].notna() & (df['label'] != "")]  # 去除空标签

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    train_dataset = RoadDataset(train_df, transform=transform)
    val_dataset = RoadDataset(val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ---------- 构建模型 ----------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 预训练模型
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.labels))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---------- 训练过程 ----------
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")

    # ---------- 模型保存 ----------
    torch.save(model.state_dict(), "resnet18_road_classifier.pth")
    print("✅ 模型已保存为 resnet18_road_classifier.pth")
