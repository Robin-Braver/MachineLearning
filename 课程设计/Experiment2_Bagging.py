"""实验二：编程实现Bagging模型，以决策树桩为基学习器，在西瓜数据集3.0a上训练一个Bagging集成，并与教材图8.6进行比较。"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace

def bootstrap_sampling(dataSet):
    """
    自助采样法bootstrap_sampling：采样得到与源数据集相同大小的数据集。
    :param dataSet: 数据集
    :return:        新数据集
    """
    #np.random.seed(2222)
    n = len(dataSet)
    new_dataSet=[]
    for i in range(n):
        index = np.random.randint(0, n)
        temp_dataSet = dataSet[index]
        new_dataSet.append(temp_dataSet)
    return np.array(new_dataSet)

def calErr(dataSet, feature, threshVal, inequal):
    """
    计算决策树桩的错误率
    :param dataSet:     数据集
    :param feature:     属性index
    :param threshVal:   属性阈值
    :param inequal:     不等号
    :return:            错误率
    """
    err_sum = 0
    if inequal == 'lt':#如果认为低于阈值为好瓜
        for data in dataSet:
            if (data[feature] <= threshVal and data[-1] == -1) or (data[feature] > threshVal and data[-1] == 1):#则错误判断 = 低于阈值且为坏瓜 + 高于阈值且为好瓜
                err_sum += 1
    else:#如果认为高于阈值为好瓜
        for data in dataSet:
            if (data[feature] >= threshVal and data[-1] == -1) or (data[feature] < threshVal and data[-1] == 1):#则错误判断 = 高于阈值且为坏瓜 + 低于阈值且为好瓜
                err_sum += 1
    return err_sum / float(len(dataSet))


def build_stump(dataSet):
    """
    构建决策树桩
    :param dataSet: 数据集
    :return:   返回储存决策树桩的字典。
    """
    m, n = dataSet.shape
    bestErr = np.inf
    stump = {}
    for i in range(n - 1):#依次处理每一个特征
        for j in range(m):#依次处理每一个样本
            threVal = dataSet[j][i]#初始化阈值
            for inequal in ['lt', 'gt']:#对于大于等于以及小于等于符号划分
                err = calErr(dataSet, i, threVal, inequal)#计算错误率
                if err < bestErr:#如果错误更低，保存划分信息
                    bestErr = err
                    stump["feature"] = i
                    stump["threshVal"] = threVal
                    stump["inequal"] = inequal
                    stump["err"] = err
    return stump


def predict(data, stump):
    """
    通过决策树桩预测数据的类别。
    :param data:    待预测的数据。
    :param stump:   决策树桩的字典。
    :return:    预测值。
    """
    if stump["inequal"] == 'lt':
        if data[stump["feature"]] <= stump["threshVal"]:
            return 1
        else:
            return -1
    else:
        if data[stump["feature"]] >= stump["threshVal"]:
            return 1
        else:
            return -1


def predictAll(dataSet, G):
    """
    基于基学习器预测所有样本
    :param dataSet: 数据集
    :param G: 通过Bagging得到的分类器列表
    :return: 预测结果
    """
    result = []
    for data in dataSet:
        predict_sum = 0
        for key in G.keys():
            predict_sum += predict(data, G[key])

        if predict_sum > 0:
            result.append(1)
        else:
            result.append(-1)

    return np.array(result)


def calcAcc(dataSet, G):
    """
    计算分类器的准确度
    :param dataSet:    数据集
    :param G:          通过Bagging得到的分类器列表
    :return:           准确度
    """
    right_sum = 0
    right_label=[]
    predict_label=[]
    for data in dataSet:
        predict_sum = 0
        for key in G.keys():
            predict_sum += predict(data, G[key])
        if (predict_sum > 0 and data[-1] == 1) or (predict_sum <= 0 and data[-1] == -1):
            right_sum += 1

        right_label.append(int(data[-1]))
        if predict_sum>0:
            predict_label.append(1)
        else:
            predict_label.append(-1)

    return right_sum / float(len(dataSet)),right_label,predict_label


def Bagging(dataSet, T):
    """
    通过bootstrap_sampling从原数据中得到新的数据，训练新的分类器。
    :param dataSet:     数据集
    :param T:           分类器数量
    :return:            分类器字典
    """
    G = {}
    for i in range(T):
        temp_dataSet = bootstrap_sampling(dataSet)
        stump = build_stump(temp_dataSet)
        G[i]=stump
    return G


def plotData(data, G):
    """
    绘制数据集
    :param data:    数据集
    :param G:       学习器字典
    :return: 
    """
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
    for key in G.keys():
        z = [G[key]["threshVal"]]*100
        if G[key]["feature"] == 0:
            plt.plot(z, y, color='gray', linestyle=':')
        else:
            plt.plot(x, z, color='gray', linestyle=':')
    accuracy,right,pred = calcAcc(datas,G)
    acc = "---正确率为：" + str(accuracy)
    plt.scatter(X1, Y1, marker='+', label='好瓜', color='k')
    plt.scatter(X2, Y2, marker='_', label='坏瓜', color='r')
    plt.xlabel('密度')
    plt.ylabel('含糖率')
    plt.xlim(0, 0.8)  # 设置x轴范围
    plt.ylim(0, 0.6)  # 设置y轴范围
    plt.title(str(len(G)) + "个基学习器"+acc)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.legend(loc='upper left')
    pltBound(G)
    plt.show()



def pltBound(G):
    """
    画分类边界
    :param G: 通过Bagging得到的分类器列表
    :return:
    """
    x_tmp = np.linspace(0, 0.8, 600)
    y_tmp = np.linspace(0, 0.6, 600)
    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)

    Z_ = predictAll(np.c_[X_tmp.ravel(), Y_tmp.ravel()], G).reshape(X_tmp.shape)
    plt.contour(X_tmp, Y_tmp, Z_, [0], colors='red', linewidths=4)


if __name__ == '__main__':
    data = pd.read_csv('data/西瓜数据集3a.txt', sep=' ').values
    data[data[:, 2] == 0, 2] = -1   # 读取数据并将0换成-1
    for t in [3, 5, 11]:    # 学习器的数量
        G = Bagging(data, t)
        print('集成学习器（字典）：', f"G{t} = {G}")
        acc,right_label,predict_label=calcAcc(data, G)
        print("准确值：",right_label)
        print("预测值：",predict_label)
        print('准确率=', acc)
        plotData(data, G)