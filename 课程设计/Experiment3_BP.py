import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from sklearn.decomposition import PCA


def sigmoid(x):
    """
    Sigmoid函数：将输入值转化为接近0或1的值
    :param x:需要处理的变量（这里多为矩阵）
    :return 经sigmoid函数处理后的值
    """
    return 1.0 / (1 + np.exp(-x))


def BP_revalue(filename):
    """
    BP_revalue:读取文件，将数据降维同时将数据标签重新赋值
    :return: 变量全为数值的变量，以及新的特征标签。
    """
    dataSet,dataLabel,col_name = utils.getData(filename)
    dataSet = pd.DataFrame(dataSet,columns=col_name[:-1])
    if '胸痛类型' in dataSet.columns:
        num_C = 3
    else:
        num_C = 2
    #将标签中的2改为0
    for i in range(len(dataLabel)):
        if dataLabel[i]==2:
            dataLabel[i]=0
    # 新的特征数据和类融合
    pca=PCA(n_components=num_C)
    dataSet=pca.fit_transform(dataSet)
    dataLabel = np.asarray(dataLabel, dtype="int").reshape(-1, 1)
    dataSet = np.concatenate((dataSet, dataLabel), axis=1)
    return dataSet


def ABP(dataSet, q=10, lr=0.1, thresh=0.001, epoch=2000, pro=False):
    """
    累计BP算法
    :param dataSet: 数据集
    :param q: 隐层神经元个数
    :param lr: 学习率
    :param thresh: 阈值；容差
    :param epoch: 迭代次数
    :param pro: 是否使用改进算法
    :param alpha: 遗忘因子
    :return: 隐层参数、阈值、输出层参数、阈值、误差列表
    """
    errHistory = []  # 记录每轮迭代的均方误差
    accHistory = []  # 记录每轮迭代的准确率
    y = dataSet[:, -1].reshape(-1, 1)
    x = dataSet[:, :-1]
    m, n = x.shape
    # 隐层参数
    v = np.random.randn(n, int(q + 1))  # 隐层参数
    w = np.random.randn(int(q + 1), 1)  # 输出层参数
    gamma = np.random.randn(1, int(q + 1))  # 隐层阈值
    out = np.random.random(1)  # 输出层阈值

    err_last = np.inf
    if pro == False:
        print("-----------ABP算法-----------")
    else:
        print("-----------改进ABP算法-----------")
    for Epoch in range(1,epoch+1):
        b = sigmoid(np.dot(x, v) - gamma)  # 隐层输出
        blr = np.dot(b, w)  # 输出层输出
        y_predict = sigmoid(blr - out)  # 预测值
        g = y_predict * (1 - y_predict) * (y - y_predict)  # 输出层神经元梯度项
        e = b * (1 - b) * np.dot(g, w.T)  # 隐层神经元梯度向

        err, acc, real_list, predict_list = calErr(dataSet, v, gamma, w, out)

        if pro==True:#使用升级的
            if err < err_last:
                lr = lr * 1.05
            elif err > err_last:
                lr = lr * 0.8
            err_last = err
        w += lr * np.dot(b.T, g)
        v += lr * np.dot(x.T, e)
        out -= lr * g.sum()
        gamma -= lr * e.sum(axis=0)

        if Epoch % 10 == 0 or Epoch == 1:
            errHistory.append(err)
            accHistory.append(acc)
        if Epoch % 100 == 0 or Epoch == 1:
            print("Epoch:{:^10}\t\tLoss:{:^10.6f}\t\tAcc:{:^10.6f}".format(Epoch, err, acc))

        if err<=thresh:#误差小于阈值
            break
    return v, gamma, w, out, errHistory, accHistory


def predict(data, v, gamma, w, out):
    """
    预测某一样本数据
    :param data: 样本数据
    :param v: 隐藏层参数
    :param gamma: 隐藏层阈值
    :param w: 输出层参数
    :param out: 输出层阈值
    :return: 预测值
    """
    b = sigmoid(np.dot(data, v) - gamma)
    blr = np.dot(b, w)
    y_predict = sigmoid(blr - out)
    return y_predict[0][0]


def calErr(dataSet, v, gamma, w, out):
    """
    计算均方误差
    :param dataSet: 数据集
    :param v: 隐藏层参数
    :param gamma: 隐藏层阈值
    :param w: 输出层参数
    :param out: 输出层阈值
    :return: 均方误差,准确率,真实值列表,预测值列表
    """
    x = dataSet[:, :-1]
    y = dataSet[:, -1]
    num = x.shape[0]
    err = 0.0
    real_list = []
    predict_list = []
    for i in range(num):
        pred = predict(x[i], v, gamma, w, out)
        err += ((y[i] - pred) ** 2) / 2.0

        real_list.append(int(y[i]))
        if pred > 0.5:
            predict_list.append(1)
        else:
            predict_list.append(0)

    acc = 0
    for i in range(len(real_list)):
        if real_list[i] == predict_list[i]:
            acc += 1
    return err / float(num), acc / float(num), real_list, predict_list


def show_result(filename,q = 50,lr=0.1,thresh=0.01,epoch=10000,test_size = 0.3):
    dataSet_name = filename.split('/')[1].split('.')[0]
    print("\033[31m------------------------"+dataSet_name+"------------------------\033[0m")
    dataSet = BP_revalue(filename)
    train_data, test_data = utils.splitDataSet1(dataSet, test_size=test_size)

    v2, gamma2, w2, out2, errHistory2, accHistory2 = ABP(train_data, q=q, lr=lr, thresh=thresh, epoch=epoch)
    v3, gamma3, w3, out3, errHistory3, accHistory3 = ABP(train_data, q=q, lr=lr, thresh=thresh, epoch=epoch, pro=True)

    plt.plot(np.arange(len(errHistory2))*10, errHistory2, 'r', label='ABP')
    plt.plot(np.arange(len(errHistory3))*10, errHistory3, 'b', label='改进ABP')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(dataSet_name+"-ABP/改进ABP训练损失变化图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.legend()
    plt.show()

    plt.plot(np.arange(len(accHistory2))*10, accHistory2, 'r', label='ABP')
    plt.plot(np.arange(len(accHistory3))*10, accHistory3, 'b', label='改进ABP')
    # plt.ylim(0, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(dataSet_name + "-ABP/改进ABP训练准确率变化图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.legend()
    plt.show()
    #ABP算法测试
    err, acc, real_list, predict_list = calErr(test_data, v2, gamma2, w2, out2)
    print("------------ABP算法------------")
    acc, p, r, f1 = utils.calAccuracy(predict_list, real_list)
    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))
    #改进BP算法测试
    err, acc, real_list, predict_list = calErr(test_data, v3, gamma3, w3, out3)
    print("------------改进ABP算法------------")
    acc, p, r, f1 = utils.calAccuracy(predict_list, real_list)
    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))


if __name__ == '__main__':
    show_result("data/haberman.data",q=40,lr=0.05 ,thresh=0.01,epoch=1000,test_size = 0.1)
    show_result("data/heart.dat", q=30, lr=0.05, thresh=0.01, epoch=1000, test_size=0.3)
