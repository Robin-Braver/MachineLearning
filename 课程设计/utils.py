import math
from collections import Counter

import pandas as pd
import numpy as np

def getData(filename):
    """
    getData函数：从文件中读取数据，转化为指定的ndarray格式
    :param filename:读取文件名称
    :returns  X：样本，格式为：每行一条数据，列表示属性 ndarray格式
    :returns  label：对应的标签 ndarray格式
    :returns column_name:列名，主要用于决策树构建
    """
    data = pd.read_csv(filename,index_col=False)
    X = data.iloc[:,0:-1]
    label = data.iloc[:,-1]
    column_name = data.columns
    return np.array(X),np.array(label),np.array(column_name)

def splitDataSet(X,Y,test_size=0.3,seed = None):
    """
    splitDataSet函数：划分数据集
    :param X:待划分样本数据
    :param Y:待划分样本标签
    :param test_size:测试集占比
    :param seed:随机数种子
    :returns testdata:测试集 train_size:训练集
    """
    if seed != None:
        np.random.seed(seed)
    total_num = len(X)
    test_num = int(total_num*test_size)
    Y = Y.reshape(len(Y),1)
    data =np.concatenate((X, Y),axis=1)
    np.random.shuffle(data)
    test_data = data[0:test_num,0:-1]
    test_label = data[0:test_num,-1:data.shape[1]]
    train_data = data[test_num:total_num,0:-1]
    train_label = data[test_num:total_num, -1:data.shape[1]]
    return train_data,train_label.reshape(len(train_label)),test_data,test_label.reshape(len(test_label))


def splitDataSet1(data, test_size=0.3,seed = 11):
    """
    splitDataSet1函数：划分数据集，是一个重载函数，这里主要对BP用到的dataset进行划分
    :param data:待划分样本
    :param test_size:测试集占比
    :param seed:随机数种子
    :returns testdata:测试集 train_size:训练集
    """
    np.random.seed(seed)
    total_num = len(data)
    test_num = int(total_num * test_size)
    index_list = list(range(0, total_num))
    np.random.shuffle(index_list)
    test_data = data[index_list[0:test_num],]
    train_data = data[index_list[test_num:total_num],]
    return train_data, test_data


def readDataSet(filename='haberman.data'):
    """
    读取数据集
    :param filename: 数据集文件名称
    :return: 数据集、标签集
    """
    if filename=='haberman.data':
        dataSet = pd.read_csv(filename, sep=',')
    else:
        dataSet = pd.read_csv(filename,sep=',')
    features = dataSet.columns
    dataSet = np.array(dataSet)

    return dataSet[:, :-1], dataSet[:, -1], features


def getDataSet(filename='haberman.data'):
    """
    利用pandas将分类变量转化为数值变量。将分类变量进行one-hot编码。
    :return: 变量全为数值的变量，以及新的特征标签。
    """
    dataSet, dataLabel, features = readDataSet(filename)
    df = pd.DataFrame(dataSet)

    #将标签中的2改为0
    for i in range(len(dataLabel)):
        if dataLabel[i]==2:
            dataLabel[i]=0

    #对数据进行归一化处理
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    dataSet=normDataSet

    dataLabel = np.asarray(dataLabel, dtype="int").reshape(-1, 1)
    dataSet = np.concatenate((dataSet, dataLabel), axis=1)

    return dataSet

def calAccuracy(pred,label):
    """
    calAccuracy函数：计算准确率
    :param pred:预测值
    :param label:真实值
    :returns acc：正确率
    :returns p：查准率
    :returns r：查全率
    :returns f1:F1度量
    """
    num = len(pred)#样本个数
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    positive = max(label)
    negetive = min(label)
    '''分别计算TP FP FN TN'''
    for i in range(num):
        if pred[i]==positive and label[i]==positive:
            TP += 1
        elif pred[i]==positive and label[i]==negetive:
            FP += 1
        elif pred[i]==negetive and label[i]==positive:
            FN += 1
        elif pred[i]==negetive and label[i]==negetive:
            TN += 1
    '''计算准确率 查准率 查全率 F1度量'''
    acc = (TN+TP) / num
    if TP+FP==0:
        p = float("inf")
    else:
        p = TP/(TP+FP)
    if TP+FN == 0:
        r = float("inf")
    else:
        r = TP/(TP+FN)
    if p+r==0:
        f1 = float("inf")
    else:
        f1 = 2*p*r/(p+r)
    return acc,p,r,f1

def cal_ent(labels):
    """
    计算信息熵
    :param labels:样本标签
    :return:float 信息熵
    """
    class_dict = Counter(labels)
    num = len(labels)
    # 计算信息熵
    ent = 0
    for key in class_dict.keys():
        p = float(class_dict[key]) / num
        ent -= p * math.log(p, 2)
    return ent

def cal_gainRatio(data,label,col_name):
    """
    计算各属性的信息增益率
    :param: data:样本数据
    :param: label:标签
    :param: col_name:列名
    :return: gainRatio_list各个属性的信息增益率
    """
    gainRatio_list = []  # 记录各个属性的增益率
    gain_list = []
    ent_all = cal_ent(label)#所有样本的ent
    for index in range(data.shape[1]):#遍历每个属性求信息增益
        new_ent_ratio = 0#当前样本属性的信息增益率
        feature_list = data[:,index] # 提取某一列的值
        if col_name[index] not in ['性别', '胸痛类型', '血糖', '心电图结果', '心绞痛', 'ST斜率', '贫血']:#代表是连续值
            sorted_feature_list = sorted(feature_list)  # 对连续值排序
            maxGain = -999
            for k in range(len(data) - 1):  # 考察选取哪一个中位数点
                mid_feature = (sorted_feature_list[k] + sorted_feature_list[k + 1]) / 2  # 计算中位数
                left_label = [label[i] for i in range(len(data)) if data[i][index] <= mid_feature] # 提取属性值小于该取值的标签
                right_label = [label[i] for i in range(len(data)) if data[i][index] > mid_feature]#同理提取大于该取值的标签

                p_left = len(left_label) / len(label)
                p_right = len(right_label) / len(label)
                tmp_ent = ent_all - p_left * cal_ent(left_label) - p_right *cal_ent(right_label)  # 划分后只有两个类别
                if tmp_ent > maxGain:
                    maxGain = tmp_ent
                    IV = -p_left * math.log(p_left, 2) - math.log(p_right, 2) * p_right
                    Gain_ratio = maxGain / IV   # 计算信息增益
            gain_list.append(maxGain)
            gainRatio_list.append(Gain_ratio)
        else:
            unique_feature = np.unique(feature_list)#找出出现了哪几个值
            IV = 0
            for feature in unique_feature:
                tmp_label = [label[i] for i in range(len(data)) if data[i][index]==feature]#提取属性值为该取值的标签
                new_ent_ratio += len(tmp_label)/len(label)*cal_ent(tmp_label)
                IV -= len(tmp_label)/len(label) * math.log(len(tmp_label)/len(label), 2)
            new_ent_ratio = ent_all - new_ent_ratio  # 计算信息增益
            gain_list.append(new_ent_ratio)
            new_ent_ratio = new_ent_ratio / IV  # 计算增益率
            gainRatio_list.append(new_ent_ratio)
    # av_gain = np.average(gain_list)  # 求均值
    # for i in range(len(gain_list)):
    #     if gain_list[i]>=av_gain:
    #         gainRatio_list[i] = gainRatio_list[i]*10
    #     else :
    #         gainRatio_list[i] = gainRatio_list[i]*0.1
    return gainRatio_list



if __name__=='__main__':
    # data = pd.read_csv("data/haberman.data")
    # data = pd.read_csv("data/heart.dat")
    # for i in range(data.shape[1]):
    #     print(i,np.unique(data.iloc[:,i]))

    X,Y ,col_name= getData("data/haberman.data")
    X,Y ,col_name= getData("table4.2.txt")
    #X, Y,_ = getData("data/heart.dat")
    #train_data, train_label, test_data, test_label = splitDataSet(X,Y,test_size=0.3)
    gainRatio_list = cal_gainRatio(X,Y,col_name[:-1])
    print(gainRatio_list)
