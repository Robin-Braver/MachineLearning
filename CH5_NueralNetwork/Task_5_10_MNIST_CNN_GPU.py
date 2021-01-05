'''自学并编程实现一个卷积神经网络，在手写字符识别数据集MNIST上进行实验测试，
并对代码结构进行理解和分析'''
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import time

EPOCH = 10
BATCH_SIZE = 2048
LR = 0.001#学习率
DOWNLOAD_MNIST = True#是否下载数据集

#获取数据集
train_data = torchvision.datasets.MNIST(root='./mnist/', train=True,
                                        transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
# 分批装载到train_loader
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 测试样本
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False,transform=torchvision.transforms.ToTensor())

valid_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:1000].cuda()/255
valid_y = test_data.targets[:1000].cuda()
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2),)
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),)
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

cnn = CNN().cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# 训练
start=time.perf_counter()#计时间
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = x.cuda()
        b_y = y.cuda()
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            test_output = cnn(valid_x)
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
            accuracy = float((pred_y == valid_y.data.cpu().numpy()).astype(int).sum()) / float(valid_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| valid accuracy: %.2f' % accuracy)

print("------------------------------------------------")
correct = 0
for step, (x, y) in enumerate(test_loader):
    b_x = x.cuda()
    b_y = y.cuda()
    output = cnn(b_x)
    loss = loss_func(output, b_y)
    pred_y = torch.max(output, 1)[1].data.cpu().numpy()
    correct += (pred_y == b_y.data.cpu().numpy()).astype(int).sum()
print('| test loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.4f' % (correct/len(test_loader.dataset)))

print("------------------------------------------------")
test_output = cnn(valid_x[:20])
pred_y = torch.max(test_output, 1)[1].cuda().data
dur=time.perf_counter()-start
print('测试集中前20个数字预测值：',pred_y)
print('测试集中前20个数字真实值：',valid_y[:20])
print("花费时间为{:.2f}".format(dur))
