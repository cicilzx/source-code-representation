import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

# LSTM的训练和预测，Devign

feature_len = 64
INPUT_SIZE = 64
BATCH_SIZE = 64
LR = 0.01               # learning rate


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


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=100,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(100, 2)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, h_c = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])
        return out


if __name__ == '__main__':
    rnn = RNN()
    rnn.cuda()
    print(rnn)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    train_data, test_data, train_label, test_label = \
        get_data('F:/iSE/Source Code representation/CODE/OJ-clone/training/syntax_semantic.csv')
    # train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    # training and testing
    for epoch in range(10000):
        train_data = train_data.view(-1, 1, 64)
        out = rnn(train_data.cuda())
        loss = loss_func(out, train_label.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            prediction = torch.max(out, 1)[1].cuda()
            pred_y = prediction.data.cpu().numpy().squeeze()
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
                    torch.save(rnn.state_dict(), 'model/' + str(F1) + 'checkpoint.pth')

    torch.save(rnn.state_dict(), 'checkpoint.pth')
    test_data = test_data.view(-1, 1, 64)
    out = rnn(test_data.cuda())
    prediction = torch.max(out, 1)[1].cuda()

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