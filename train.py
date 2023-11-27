import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import os
import time


# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(13 * 13, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 13 * 13)
        x = self.fc(x)
        return x

script_dir = os.path.dirname(__file__)  # 获取脚本所在的目录

# 数据预处理
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
transform = transforms.Compose([transforms.ToTensor()])

# 加载数据集
trainset = torchvision.datasets.FashionMNIST(os.path.join(script_dir, '../../data'), download=True, train=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(os.path.join(script_dir, '../../data'), download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 创建模型
model = LeNet()
model = model.to('cuda')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()

# 训练模型
for epoch in range(100):
    # print('epoch ', epoch)
    for inputs, labels in trainloader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

end_time = time.time()
elapsed_time = end_time - start_time
rounded_time = round(elapsed_time, 4)
print(f"{rounded_time}")

# 测试模型
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images, labels = images.to('cuda'), labels.to('cuda')
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
        
# print(correct/total)  

# 导出模型参数，也可以自定义导出模型参数的文件格式，这里使用了最简单的方法，但请注意，如果改动了必须保证程序二能够正常读取
for name, param in model.named_parameters():
    np.savetxt(os.path.join(script_dir, f'./{name}.txt'), param.detach().cpu().numpy().flatten())

