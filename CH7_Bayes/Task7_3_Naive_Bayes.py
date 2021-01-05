"""7.3 （1）基本的朴素贝叶斯分类，以西瓜数据集3.0为训练集，对“测1”样本进行判别。
7.3（2）实现拉普拉斯修正的朴素贝叶斯分类器，并和7.3（1）做对比实验"""
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def train(data, label,is_Laplacian=True):
    """
    train：训练函数，根据训练样本计算类先验概率以及各属性条件概率
    :param data:训练数据
    :param label: 训练标签
    :param is_Laplacian:是否需要拉普拉斯修正，默认带修正
    :returns:p1：好瓜先验概率
    :returns:px1_list:好瓜中每个属性的条件概率，是一个二维列表，离散值直接返回而连续值返回的方差和均值
    :returns:px0_list:坏瓜中每个属性的条件概率，是一个二维列表，离散值直接返回而连续值返回的方差和均值
    """
    num_sample, num_attr = data.shape#样本数量，属性数量
    N = len(np.unique(label)) #样本中类别数目，在此例N=2

    #类先验概率
    if is_Laplacian:#带修正
        p1 = (len(label[label == 1])+1)/(num_sample +N) #正样本概率
    else:
        p1 = (len(label[label == 1]))/(num_sample)
    #各属性的条件概率
    px1_list = []
    px0_list = []

    data1 = data[label == 1]#分别提取好瓜和坏瓜
    data0 = data[label == 0]
    num_sample1 = len(data1)#好瓜坏瓜个数
    num_sample0 = len(data0)

    for i in range(num_attr):#依次处理每个属性
        xi = data.iloc[:, i]#提取该列属性值
        is_continuous = False #判断值是否是连续值
        if data.columns[i] in ['密度','含糖率']:
            is_continuous = True
        xi1 = data1.iloc[:, i]
        xi0 = data0.iloc[:, i]
        if is_continuous:  # 处理连续值，不需要考虑拉普拉斯修正
            xi1_mean = np.mean(xi1)
            xi1_var = np.var(xi1)
            xi0_mean = np.mean(xi0)
            xi0_var = np.var(xi0)
            #对于连续属性，添加标志，平均值，方差
            px1_list.append([is_continuous,xi1_mean,xi1_var])
            px0_list.append([is_continuous, xi0_mean, xi0_var])
        else: #处理离散值，需要分别处理带和不带的情况
            unique_value = xi.unique()  # 当前取值情况
            if is_Laplacian:#带修正
                num_value = len(unique_value)  # 当前属性取值个数
                # 计算样本中，该属性每个取值的数量，并且加1，如果没有取值就设置为0后在加1
                xi1_value_count = pd.value_counts(xi1)[unique_value].fillna(0) + 1
                xi0_value_count = pd.value_counts(xi0)[unique_value].fillna(0) + 1
                #计算条件概率
                p_xi1_c = np.log(xi1_value_count /(num_sample1 + num_value))
                p_xi0_c = np.log(xi0_value_count / (num_sample0 + num_value))
            else:#不带修正
                # 计算样本中，该属性每个取值的数量，并且加1，如果没有取值就设置为0后在加1
                xi1_value_count = pd.value_counts(xi1)[unique_value].fillna(0)
                xi0_value_count = pd.value_counts(xi0)[unique_value].fillna(0)
                # 计算条件概率
                p_xi1_c = np.log(xi1_value_count / (num_sample1))#可能存在log里面为0，此时取-inf
                p_xi0_c = np.log(xi0_value_count / (num_sample0))
            #对于离散属性，添加标志，条件概率
            px1_list.append([is_continuous,p_xi1_c])
            px0_list.append([is_continuous,p_xi0_c])
    return p1, px1_list, px0_list


def predict_Onesample(data,p1, p1_list, p0_list):
    """
    predict_Onesample: 根据参数预测1个样本数据的取值
    :param:data:测试数据
    :param:p1：好瓜先验概率
    :param:px1_list:好瓜中每个属性的条件概率，是一个二维列表，离散值直接返回而连续值返回的方差和均值
    :param:px0_list:坏瓜中每个属性的条件概率，是一个二维列表，离散值直接返回而连续值返回的方差和均值
    :return 1or0:好瓜或坏瓜
    """
    num_attr = len(data)#属性数量
    #好瓜坏瓜概率取对数，防止下溢
    x_p1 = np.log(p1)
    x_p0 = np.log(1-p1)

    for i in range(num_attr):
        p1_xi = p1_list[i]
        p0_xi = p0_list[i]
        if p1_xi[0]==True:#如果数据是连续值,则求出值
            mean1 = p1_xi[1]
            var1 = p1_xi[2]
            mean0 = p0_xi[1]
            var0 = p0_xi[2]
            x_p1 += np.log(1 / (np.sqrt(2 * np.pi * var1)) * np.exp(- (data[i] - mean1) ** 2 / (2 * var1)))
            x_p0 += np.log(1 / (np.sqrt(2 * np.pi * var0)) * np.exp(- (data[i] - mean0) ** 2 / (2 * var0)))
        else:#离散属性直接读取出改取值的概率
            x_p1 += p1_xi[1][data[i]]
            x_p0 += p0_xi[1][data[i]]
    if x_p1 > x_p0:
        return 1
    else:
        return 0

def predict(data,p1, p1_list, p0_list):
    """
    predict: 根据参数预测多个样本数据的取值
    :param:data:测试数据
    :param:p1：好瓜先验概率
    :param:px1_list:好瓜中每个属性的条件概率，是一个二维列表，离散值直接返回而连续值返回的方差和均值
    :param:px0_list:坏瓜中每个属性的条件概率，是一个二维列表，离散值直接返回而连续值返回的方差和均值
    :return pred:预测值
    """
    num_sample = len(data)#属性数量
    pred = []

    for i in range(num_sample) :#对每个样本做预测
        x = data.iloc[i, :]
        pred_i = predict_Onesample(x,p1, p1_list, p0_list)
        pred.append(pred_i)
    return pred


if __name__ == '__main__':
    data = pd.read_csv('西瓜数据集4.3.txt')
    data_one = data.iloc[0,:-1]
    data = shuffle(data).reset_index(drop=True)
    train_data = data.iloc[0:12, :-1]
    train_label = data.iloc[0:12, -1]
    test_data = data.iloc[12:, :-1].reset_index(drop=True)
    test_label = data.iloc[12:,-1].reset_index(drop=True)

    p1, px1_list, px0_list = train(train_data,train_label,is_Laplacian=False)#不带拉普拉斯变换
    Lp1, Lpx1_list, Lpx0_list = train(train_data, train_label, is_Laplacian=True) #带拉普拉斯变换

    print("---------对第一个样本的预测情况--------------")
    print("不带拉普拉斯变换的预测值：",predict_Onesample(data_one,p1, px1_list, px0_list))
    print("带拉普拉斯变换的预测值：", predict_Onesample(data_one,Lp1, Lpx1_list, Lpx0_list))
    print("---------对所有样本的预测情况--------------")
    test_label = test_label.tolist()
    pred =  predict(test_data, p1, px1_list, px0_list)
    Lpred = predict(test_data, Lp1, Lpx1_list, Lpx0_list)
    correct = 0
    Lcorrect = 0
    for i in range(len(test_label)):
        if(pred[i]==test_label[i]):
            correct += 1
        if (Lpred[i] == test_label[i]):
            Lcorrect += 1
    print("样本真实标签")
    print(test_label)
    print("不带拉普拉斯变换的预测值：{},正确率：{}/{}={}".format(pred,correct,len(test_label),1.0*correct/len(test_label)))
    print("带拉普拉斯变换的预测值：{},正确率：{}/{}={}".format(Lpred, Lcorrect, len(test_label), 1.0 * Lcorrect / len(test_label)))