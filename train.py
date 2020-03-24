import torch as th
import torchvision as tv
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch import optim

show = ToPILImage()
# 数据预处理
tranform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
# 训练集
trainset = tv.datasets.CIFAR10(root='D:/workspace/code/pytorch/CIFAR10/', train=True, download=False,
                               transform=tranform)
trainloader = th.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
# 测试集
testset = tv.datasets.CIFAR10(root='D:/workspace/code/pytorch/CIFAR10/', train=True, download=False, transform=tranform)
testloader = th.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print('Files already downloaded an verified')

(data, label) = trainset[100]
print(classes[label])
show(data + 1 / 2)
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(''.join('%11s' % classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images + 1) / 2)).resize((400, 100))

#####定义网络
import torch.nn as nn
import torch.nn.functional as tfn


class Net(nn.Module):
    def __init__(self):
        print('step __init__')
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = tfn.max_pool2d(tfn.relu(self.conv1(x)), (2, 2))

        x = tfn.max_pool2d(tfn.relu(self.conv2(x)), 2)

        x = x.view(x.size()[0], -1)

        x = tfn.relu(self.fc1(x))

        x = tfn.relu(self.fc2(x))

        x = self.fc3(x)
        return x


net = Net()
print(net)
#####定义优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for eporch in range(20):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()

        # 向前、向后传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d,%5d] loss:%.3f' % (eporch + 1, i + 1, running_loss / 2000))
            running_loss = 0
    print('Finished Trainning')

dataiter = iter(testloader)
images, labels = dataiter.next()
#####实际labels
print('实际labels:', ' '.join('%08s' % classes[labels[j]] for j in range(4)))
# show(tv.utils.make_grid(images/2-0.5)).resize((400,100))
show(tv.utils.make_grid(images / 2 - 0.5)).resize((400, 100))

#####计算预测网络的label
output = net(Variable(images))
_, predicted = th.max(output.data, 1)
print('预测结果',
      ' '.join('%5s' % classes[predicted[j]] for j in range(4)))  # .join('%5s'%classes[predicted[j]] for in range(4))

#####整个数据集测试
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = th.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('10000张测试集集中的准确率为 %d,%%' % (100 * correct / total))
