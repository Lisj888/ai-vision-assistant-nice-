import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import gradio as gr
import numpy as np


# 1. 定义 CNN（卷积神经网络）—— 这才是真正的图像识别！
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 2. 准备数据（加上数据增强，让它能应付乱七八糟的手写）
transform = transforms.Compose([
    transforms.RandomRotation(15),  # 随机旋转15度
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
    transforms.ToTensor()
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("正在训练 CNN 模型（大概需要2-3分钟）...")
for epoch in range(2):  # 训练两轮，效果更好
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 300 == 0:
            print(f"Epoch {epoch + 1}, 进度：{i}/938，Loss：{loss.item():.4f}")
print("模型训练完成！")


# 3. 预测函数
def predict(img, invert):
    if isinstance(img, dict):
        img = Image.open(img['composite']['path'])
    img = img.convert('L')
    img = img.resize((28, 28))
    if invert:
        img = Image.eval(img, lambda x: 255 - x)
    img = img.point(lambda x: 255 if x > 128 else 0)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return f"AI 识别结果：{predicted.item()}"


# 4. 网页界面
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Sketchpad(label="用鼠标在黑色画板上写白色数字（0-9）", type="pil"),
        gr.Checkbox(label="反色（如果画板是白底黑字才勾选）", value=False)
    ],
    outputs=gr.Textbox(label="识别结果"),
    title="手写数字识别器（CNN终极版）",
    description="这是用CNN训练的模型，能适应更复杂的手写风格。"
)

# ====== 先验证模型本身 ======
from torchvision import datasets, transforms
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

print("\n=== 拿MNIST自己的图测试 ===")
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        output = model(images)
        _, predicted = torch.max(output, 1)
        print(f"正确答案：{labels.item()}，模型猜：{predicted.item()}")
        if i >= 9:
            break

# ====== 再打开网页 ======
interface.launch()