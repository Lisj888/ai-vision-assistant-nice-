import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. 准备数据（自动下载MNIST手写数字数据集）
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# 2. 定义一个简单的神经网络（别怕，就三层）
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 输入是28x28像素，展平成784个数字
        self.fc2 = nn.Linear(128, 10)     # 输出10个数字（0到9）

    def forward(self, x):
        x = x.view(-1, 28*28)             # 把图片展平
        x = torch.relu(self.fc1(x))       # 第一层 + 激活函数
        x = self.fc2(x)                   # 第二层输出
        return x

model = SimpleNet()
criterion = nn.CrossEntropyLoss()         # 损失函数（用来衡量错得多离谱）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器（用来改错）

# 3. 开始训练（只训练1轮，看看效果）
for epoch in range(1):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"训练进度：{i}/938，当前损失：{loss.item():.4f}")

print("训练完成！")

# 4. 测试准确率
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试集准确率：{100 * correct / total:.2f}%")