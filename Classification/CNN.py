import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

BATCH_SIZE = 50000
EPOCHS = 2
##仅使用CPU训练
DEVICE = torch.device("cpu")

#加载数据集

#训练集
train_loader =  torch.utils.data.DataLoader(
    #本地加载数据集，不用下载
    datasets.MNIST('../data',train=True,download=False,
                   transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = BATCH_SIZE, shuffle = True
)

#测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data',train=True,download=False,
                   transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = BATCH_SIZE, shuffle = True
)



#定义神经网络
class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        #卷积层
        self.conv1 = nn.Conv2d(1,10,5)
        self.conv2 = nn.Conv2d(10,20,3)

        #全连接层
        self.fc1 = nn.Linear(20*10*10,500)
        self.fc2 = nn.Linear(500,10)

    def forward(self,x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)    #10*12*12
        # out = out.view(in_size, -1)
        # out = self.fc1(out)
        #第一层结束
        out = self.conv2(out)
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        # 第二层结束
        out = F.log_softmax(out, dim=1)
        return out


#实例化网络
MNIST_myNet = myNet().to(DEVICE)   #公平起见，仅使用CPU
opt = optim.Adam(MNIST_myNet.parameters())


#定义训练函数
def train(MNIST_myNet,device,train_loader,optimizer,epoch):
    MNIST_myNet.train()
    for batch_index,(data,target), in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = MNIST_myNet(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if (batch_index + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                       100. * batch_index / len(train_loader), loss.item()))

#定义测试函数
def test(MNIST_myNet, device, test_loader):
    MNIST_myNet.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = MNIST_myNet(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



#训练
for epoech in range(1,EPOCHS+1):
    train(MNIST_myNet,DEVICE,train_loader,opt,epoech)
    test(MNIST_myNet,DEVICE,test_loader)