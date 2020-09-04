import time

import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义超参数
batch_size = 64
learning_rate = 1e-3
num_epochs = 100
feature_len = 32


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


# 定义 Logistic Regression 模型
class Logistic_Regression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Logistic_Regression, self).__init__()
        self.logistic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        out = self.logistic(x)
        return out


if __name__ == '__main__':
    model = Logistic_Regression(32, 2)  # 输入数据大小是32 * 1
    # 定义loss和optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_data, test_data, train_label, test_label = \
        get_data('F:\\iSE\\Source Code representation\\CODE\\devign-vulnerability\\fusion.csv')

    for epoch in range(1000):
        # train_data = train_data.view(-1, 1, 32)
        # 向前传播
        out = model(train_data)
        loss = criterion(out, train_label)
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            prediction = torch.max(out, 1)[1]
            pred_y = prediction.data.cpu().numpy()
            target_y = train_label.data.cpu().numpy()
            accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)

            tp = 0
            tn = 0
            fn = 0
            fp = 0
            for pred_id in range(len(pred_y)):
                pred_cur = pred_y[pred_id]
                target_cur = target_y[pred_id]
                if (1 == pred_cur) and (1 == target_cur):
                    tp += 1
                if (0 == pred_cur) and (0 == target_cur):
                    tn += 1
                if (0 == pred_cur) and (1 == target_cur):
                    fn += 1
                if (1 == pred_cur) and (0 == target_cur):
                    fp += 1
            if tp + fp != 0 and tp + fn != 0:
                precision = tp / float(tp + fp)
                recall = tp / float(tp + fn)
                F1 = 2 * precision * recall / (precision + recall)
                print("epoch=", epoch, "precision=", precision, "recall=", recall, "F1=", F1, "accuracy=", accuracy)
                if F1 > 0.9:
                    torch.save(model.state_dict(), 'model/' + str(F1) + 'checkpoint.pth')

    torch.save(model.state_dict(), 'checkpoint.pth')
    # test_data = test_data.view(-1, 1, 32)
    out = model(test_data)
    prediction = torch.max(out, 1)[1]

    pred_y = prediction.data.cpu().numpy()
    target_y = test_label.data.cpu().numpy()

    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)

    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for pred_id in range(len(pred_y)):
        pred_cur = pred_y[pred_id]
        target_cur = target_y[pred_id]
        if (1 == pred_cur) and (1 == target_cur):
            tp += 1
        if (0 == pred_cur) and (0 == target_cur):
            tn += 1
        if (0 == pred_cur) and (1 == target_cur):
            fn += 1
        if (1 == pred_cur) and (0 == target_cur):
            fp += 1

    if tp + fp != 0 and tp + fn != 0:
        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        F1 = 2 * precision * recall / (precision + recall)
        print("precision=", precision, "recall=", recall, "F1=", F1, "accuracy=", accuracy)
