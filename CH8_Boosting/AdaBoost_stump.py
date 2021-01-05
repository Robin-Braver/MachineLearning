import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace

def calErr(dataSet, feature, threshVal, inequal, D):
    """
    计算数据带权值的错误率。
    :param dataSet:[密度，含糖量，好瓜]
    :param feature: 密度，含糖量]
    :param threshVal:
    :param inequal: 'lt' or 'gt. (大于或小于）
    :param D:  数据的权重。错误分类的数据权重会大。
    :return:错误率。
    """
    DFlatten = D.flatten()   # 变为一维
    errCnt = 0
    i = 0
    if inequal == 'lt':  #如果认为低于阈值为好瓜
        for data in dataSet:
            if (data[feature] <= threshVal and data[-1] == -1) or \
               (data[feature] > threshVal and data[-1] == 1):  #则错误判断 = 低于阈值且为坏瓜 + 高于阈值且为好瓜
                errCnt += 1 * DFlatten[i]  #该样本的权重作为错误率
            i += 1
    else:
        for data in dataSet:
            if (data[feature] >= threshVal and data[-1] == -1) or \
               (data[feature] < threshVal and data[-1] == 1):
                errCnt += 1 * DFlatten[i]
            i += 1
    return errCnt


def buildTree(dataSet, D):
    """
    通过带权重的数据，建立错误率最小的决策树。
    :param dataSet:
    :param D:
    :return: 建立好的决策树的信息,如feature，threshVal，inequal，err。
    """
    m, n = dataSet.shape
    bestErr = np.inf
    Tree = {}
    numSteps = 16.0  # 每个特征迭代的步数
    for i in range(n-1):#依次处理每一个特征
        rangeMin = dataSet[:, i].min()
        rangeMax = dataSet[:, i].max()#每个属性列的最大最小值
        stepSize = (rangeMax - rangeMin) / numSteps  # 每一步的长度
        for j in range(m):#依次处理每一个样本
            threVal = rangeMin + float(j) * stepSize# 每一步划分的阈值
            for inequal in ['lt', 'gt']:    # 对于大于或等于符号划分。
                err = calErr(dataSet, i, threVal, inequal, D)  # 错误率
                if err < bestErr:  # 如果错误更低，保存划分信息。
                    bestErr = err
                    Tree["feature"] = i
                    Tree["threshVal"] = threVal
                    Tree["inequal"] = inequal
                    Tree["err"] = err
    return Tree


def predict(data, bestStump):
    """
    通过决策树桩预测数据
    :param data:        待预测数据
    :param bestStump:   决策树桩。
    :return:
    """
    if bestStump["inequal"] == 'lt':
        if data[bestStump["feature"]] <= bestStump["threshVal"]:
            return 1
        else:
            return -1
    else:
        if data[bestStump["feature"]] >= bestStump["threshVal"]:
            return 1
        else:
            return -1


def AdaBoost(dataSet, num_classifier):
    """
    训练多个分类器
    :param dataSet: 数据集。
    :param num_classifier:迭代次数，即训练多少个分类器
    :return: 字典，包含了多个分类器详情。
    """
    m, n = dataSet.shape
    D = np.ones((1, m))/m # 初始化权重，每个样本的初始权重是相同的。
    classLabel = dataSet[:, -1].reshape(1, -1) # 提取标签数据。
    G = {}  # 保存分类器的字典，分别保存权重、树等信息

    for t in range(num_classifier):
        tree = buildTree(dataSet, D)# 根据样本权重D建立一个决策树
        err = tree["err"]
        alpha = np.log((1 - err) / err) / 2# 第t个分类器的权值
        # 更新训练数据集的权值分布
        pre = np.zeros((1, m))
        for i in range(m):
            pre[0][i] = predict(dataSet[i], tree)
        a = np.exp(-alpha * classLabel * pre)
        D = D * a / np.dot(D, a.T)

        G[t] = {}
        G[t]["alpha"] = alpha
        G[t]["tree"] = tree
    return G


def adaPredic(data, G):
    """
    通过Adaboost得到的总的分类器来进行分类。
    :param data:待分类数据。
    :param G:字典，包含了多个决策树桩
    :return:预测值
    """
    score = 0
    for key in G.keys():
        pre = predict(data, G[key]["tree"])  #每个基分类器的预测结果
        score += G[key]["alpha"] * pre        #加权结合后的集成预测结果
    flag = 0
    if score > 0:
        flag = 1
    else:
        flag = -1
    return flag


def calcAcc(dataSet, G):
    """
    计算准确度
    :param dataSet:     数据集
    :param G:           字典，包含了多个决策树桩
    :return:
    """
    rightCnt = 0
    for data in dataSet:
        pre = adaPredic(data, G)
        if pre == data[-1]:
            rightCnt += 1
    return rightCnt / float(len(dataSet))


# 绘制数据集，clf为获得的集成学习器
def plotData(data, clf):
    X1, X2 = [], []
    Y1, Y2 = [], []
    datas=data
    labels=data[:,2]
    for data, label in zip(datas, labels):#分别提取好瓜和坏瓜
        if label > 0:
            X1.append(data[0])
            Y1.append(data[1])
        else:
            X2.append(data[0])
            Y2.append(data[1])
    x = linspace(0, 0.8, 100)
    y = linspace(0, 0.6, 100)
    for key in clf.keys():
        z = [clf[key]["tree"]["threshVal"]]*100
        if clf[key]["tree"]["feature"] == 0:
            plt.plot(z, y)
        else:
            plt.plot(x, z)

    plt.scatter(X1, Y1, marker='+', label='好瓜', color='k')
    plt.scatter(X2, Y2, marker='_', label='坏瓜', color='r')
    plt.xlabel('密度')
    plt.ylabel('含糖率')
    plt.xlim(0, 0.8)  # 设置x轴范围
    plt.ylim(0, 0.6)  # 设置y轴范围
    plt.title(str(len(clf)) + "个基学习器")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.legend(loc='upper left')
    plt.show()


def main():
    data = pd.read_csv('西瓜数据集3a.txt', sep=' ').values
    data[data[:,2]==0,2] = -1#读取数据并将0换成-1
    for t in [3, 5, 11]:   # 学习器的数量
        G = AdaBoost(data, t)
        print('集成学习器（字典）：',f"G{t} = {G}")
        print('准确率=',calcAcc(data, G))
        #绘图函数
        plotData(data,G)
if __name__ == '__main__':
    main()

