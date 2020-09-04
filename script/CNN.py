import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from logger import Logger
import pandas as pd
import numpy as np

# 定义超参数
batch_size = 128
learning_rate = 1e-2
num_epoches = 20
feature_len = 32


def to_np(x):
    return x.cpu().data.numpy()


def get_data(csv_path):
    csv_data = pd.read_csv(csv_path, sep=',', header=None)
    data = csv_data.values.astype(np.float)[:, 0:feature_len]
    label = csv_data.values.astype(np.int)[:, feature_len:]

    total_data = np.hstack((data, label))

    np.random.shuffle(total_data)

    total_size = len(total_data)
    train_size = int(0.8 * total_size)
    # test_size = total_size - train_size

    train_data = torch.from_numpy(total_data[0:train_size, :-1]).float()
    test_data = torch.from_numpy(total_data[train_size:, :-1]).float()
    train_label = torch.from_numpy(total_data[0:train_size, -1]).long()
    test_label = torch.from_numpy(total_data[train_size:, -1]).long()
    # test_label = torch.from_numpy(total_data[train_size:, -1].reshape(-1, 1)).int()

    return train_data, test_data, train_label, test_label


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义 Convolution Network 模型
class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True), nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential(
            nn.Linear(400, 120), nn.Linear(120, 84), nn.Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = Cnn(1, 10)  # 图片大小是28x28
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
logger = Logger('./logs')
# 开始训练
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img)
        label = Variable(label)
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.data[0]
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ========================= Log ======================
        step = epoch * len(train_loader) + i
        # (1) Log the scalar values
        info = {'loss': loss.data[0], 'accuracy': accuracy.data[0]}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), step)
            logger.histo_summary(tag + '/grad', to_np(value.grad), step)

        # (3) Log the images
        info = {'images': to_np(img.view(-1, 28, 28)[:10])}

        for tag, images in info.items():
            logger.image_summary(tag, images, step)
        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        if use_gpu:
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print()

# 保存模型
torch.save(model.state_dict(), './cnn.pth')