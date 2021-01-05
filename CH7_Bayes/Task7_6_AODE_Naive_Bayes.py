"""7.6 尝试编程实现AODE半朴素贝叶斯分类器，以西瓜数据集3.0为例，对测1样本进行判别"""
import numpy as np
import pandas as pd

def AODE_Predict_Onesample(train_data, train_label, test_data):
    """
    AODE_Predict_Onesample:根据训练集数据使用AODE半朴素贝叶斯分类器预测1个样本数据的取值
    :param train_data: 训练数据
    :param train_label: 训练标签
    :param test_data: 测试数据
    :return: pred: 预测值1or0,好瓜或坏瓜
    """
    num_sample, num_attr = train_data.shape#样本数量，属性数量
    num_attr = num_attr - 2#特征不取连续值的属性，如密度和含糖量。

    p1 = 0.0#正样本概率
    p0 = 0.0#负样本概率
    data1 = train_data[train_label == 1] #抽出类别为1的数据
    data0 = train_data[train_label == 0] #抽出类别为0的数据

    for i in range(num_attr):  # 对于第i个特征
        xi = test_data[i]
        #计算1类，第i个属性上取值为xi的样本对总数据集的占比
        Dcxi = data1[data1.iloc[:, i] == xi]  # 第i个属性上取值为xi的样本数
        Ni = len(np.unique(train_data.iloc[:,i]))
        Pcxi = (len(Dcxi) + 1) / float(num_sample + Ni)
        #计算类别为c且在第i和第j个属性上分别为xi和xj的样本，对于类别为c属性为xi的样本的占比
        mulPxjcxi = 1
        for j in range(num_attr):
            xj = test_data[j]
            Dcxixj = Dcxi[Dcxi.iloc[:, j] == xj]
            Nj = len(np.unique(train_data.iloc[:,j]))
            Pxjcxi = (len(Dcxixj) + 1) / float(len(Dcxi) + Nj)
            mulPxjcxi *= Pxjcxi
        p1 += Pcxi * mulPxjcxi

    for i in range(num_attr):  # 对于第i个特征
        xi = test_data[i]
        # 计算0类，第i个属性上取值为xi的样本对总数据集的占比
        Dcxi = data0[data0.iloc[:, i] == xi]  # 第i个属性上取值为xi的样本数
        Ni = len(np.unique(train_data.iloc[:,i]))  # 第i个属性可能的取值数
        Pcxi = (len(Dcxi) + 1) / float(num_sample + 2 * Ni)
        # 计算类别为c且在第i和第j个属性上分别为xi和xj的样本，对于类别为c属性为xi的样本的占比
        mulPxjcxi = 1
        for j in range(num_attr):
            xj = test_data[j]
            Dcxixj = Dcxi[Dcxi.iloc[:, j] == xj]
            Nj = len(np.unique(train_data.iloc[:, j]))
            Pxjcxi = (len(Dcxixj) + 1) / float(len(Dcxi) + Nj)
            mulPxjcxi *= Pxjcxi
        p0 += Pcxi * mulPxjcxi

    if p1 > p0:
        pred = 1
    else:
        pred = 0
    return pred

def AODE_Predict(train_data, train_label):
    """
    AODE_Predict:根据训练集数据使用AODE半朴素贝叶斯分类器预测所有样本数据的取值
    :param train_data: 训练数据
    :param train_label: 训练标签
    :return: pred_list: 预测值列表
    """
    pred_list=[]
    for i in range(len(train_data)):
        pred=AODE_Predict_Onesample(train_data,train_label,train_data.iloc[i,:])
        pred_list.append(pred)

    return pred_list

if __name__ == '__main__':
    data = pd.read_csv('西瓜数据集4.3.txt')
    train_data = data.iloc[:, :-1]
    train_label = data.iloc[:, -1]
    test_data = data.iloc[0, :-1]
    test_label = data.iloc[0, -1]
    print("---------对第一个样本的预测情况--------------")
    pred = AODE_Predict_Onesample(train_data, train_label, test_data)
    print("true = ", test_label)
    print("pred = ", pred)
    print("---------对所有样本的预测情况--------------")
    pred_list=AODE_Predict(train_data,train_label)
    print("true = ", list(train_label))
    print("pred = ", pred_list)
    correct = 0
    for i in range(len(train_label)):
        if (pred_list[i] == train_label[i]):
            correct += 1
    print("AODE半朴素贝叶斯分类器的预测准确率为：",round(correct/len(train_label),6))