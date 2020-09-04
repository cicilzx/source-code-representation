import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import svm
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics

feature_len = 16

# SVM text特征，Devign
def get_data(csv_path):
    csv_data = pd.read_csv(csv_path, sep=',', header=None)
    data = csv_data.values.astype(np.float)[:, 0:feature_len]
    label = csv_data.values.astype(np.int)[:, feature_len:]

    total_data = np.hstack((data, label))

    np.random.shuffle(total_data)

    total_size = len(total_data)
    train_size = int(0.8 * total_size)

    train_data = total_data[0:train_size, :-1]
    test_data = total_data[train_size:, :-1]
    train_label = total_data[0:train_size, -1]
    test_label = total_data[train_size:, -1]
    # test_label = torch.from_numpy(total_data[train_size:, -1].reshape(-1, 1)).int()

    return train_data, test_data, train_label, test_label


# SVM选择了rbf核，C选择了0.9
def svm_model(train_x, train_y, eval_x, eval_y):
    right0 = 0.0  # 记录预测为1且实际为1的结果数
    error0 = 0  # 记录预测为1但实际为0的结果数
    right1 = 0.0  # 记录预测为0且实际为0的结果数
    error1 = 0  # 记录预测为0但实际为1的结果数

    clf = svm.SVC(C=0.99, kernel='rbf')
    clf.fit(train_x, train_y)
    predicted_y = clf.predict(eval_x)

    for j in range(len(eval_x)):
        # 验证时出现四种情况分别对应四个变量存储
        pre_temp = np.array(eval_x[j]).reshape(1, 16)
        if clf.predict(pre_temp)[0] == eval_y[j] and eval_y[j] == 1:
            right0 += 1
        elif clf.predict(pre_temp)[0] == eval_y[j] and eval_y[j] == 0:
            right1 += 1
        elif clf.predict(pre_temp)[0] == 1 and eval_y[j] == 0:
            error0 += 1
        else:
            error1 += 1

    print(right0, error0, right1, error1)

    print('SVC, accuracy:', np.mean(predicted_y == eval_y))
    print("confusion_matrix:", metrics.confusion_matrix(eval_y, predicted_y))  # 混淆矩阵
    print("precision: ", metrics.precision_score(eval_y, predicted_y))
    print("recall: ", metrics.recall_score(eval_y, predicted_y))
    print("F1: ", metrics.f1_score(eval_y, predicted_y))
    print("accuracy: ", metrics.accuracy_score(eval_y, predicted_y))


if __name__ == '__main__':
    csv_path = "F:/iSE/Source Code representation/CODE/devign-vulnerability/text.csv"
    X_train, X_test, y_train, y_test = get_data(csv_path)
    svm_model(X_train, y_train, X_test, y_test)
