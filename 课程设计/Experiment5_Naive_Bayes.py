"""7.3 （1）基本的朴素贝叶斯分类，以西瓜数据集3.0为训练集，对“测1”样本进行判别。
7.3（2）实现拉普拉斯修正的朴素贝叶斯分类器，并和7.3（1）做对比实验"""
import numpy as np
import pandas as pd
import utils


def Bayes_revalue(ori_y):
    """
   Bayes_revalue函数：将y值转换为仅在0和1上取值
   :param ori_y:原始的y值
   :returns  modify_y：经过处理后的y值
   """
    max_y = max(ori_y)
    modify_y = []
    for y in ori_y:
        if y==max_y:
            modify_y.append(1)
        else:
            modify_y.append(0)
    return np.array(modify_y)

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
        if data.columns[i] not in ['性别','胸痛类型','血糖','心电图结果','心绞痛','ST斜率','贫血']:
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

def AODE_Predict_Onesample(train_data, train_label, test_data):
    """
    AODE_Predict_Onesample:根据训练集数据使用AODE半朴素贝叶斯分类器预测1个样本数据的取值
    :param train_data: 训练数据
    :param train_label: 训练标签
    :param test_data: 测试数据
    :return: pred: 预测值1or0,好瓜或坏瓜
    """
    num_sample, num_attr = train_data.shape#样本数量，属性数量

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

def AODE_Predict(train_data, train_label,test_data):
    """
    AODE_Predict:根据训练集数据使用AODE半朴素贝叶斯分类器预测所有样本数据的取值
    :param train_data: 训练数据
    :param train_label: 训练标签
    :return: pred_list: 预测值列表
    """
    pred_list=[]
    for i in range(len(test_data)):
        pred=AODE_Predict_Onesample(train_data,train_label,test_data.iloc[i,:])
        pred_list.append(pred)

    return pred_list


def SBC(train_data,train_label,valid_data,valid_label):
    """
    SBC: 改进朴素贝叶斯：基于分类精度和贪婪算法的属性选择方法，
    参考Langley P, Sage S. Induction of selective Bayesian classifiers[M]//Uncertainty Proceedings 1994. Morgan Kaufmann, 1994: 399-406.
    :param train_data:训练数据
    :param train_label: 训练标签
    :param valid_data:验证数据
    :param valid_label: 验证标签
    :returns:p1：好瓜先验概率
    :returns:px1_list:好瓜中每个属性的条件概率，是一个二维列表，离散值直接返回而连续值返回的方差和均值
    :returns:px0_list:坏瓜中每个属性的条件概率，是一个二维列表，离散值直接返回而连续值返回的方差和均值
    :returns:col_del:删除的列名
    """
    current_data = train_data #当前样本，后面会依次去掉样本
    col_name = train_data.columns.tolist()
    p1_best, px1_list_best, px0_list_best = train(train_data, train_label, is_Laplacian=True)#考虑所有样本时
    pred = predict(valid_data, p1_best, px1_list_best, px0_list_best)
    max_acc, p, r, f1 = utils.calAccuracy(pred, valid_label)
    print("当保留所有列的时候，准确率为：",max_acc)
    col_del = []#记录被删除的列
    for col in col_name:#尝试依次去掉每一列
        p1, px1_list, px0_list = train(current_data.drop(columns = [col]),train_label)
        pred = predict(valid_data.drop(columns = [col]), p1, px1_list, px0_list )
        acc,p, r, f1 = utils.calAccuracy(pred, valid_label)
        if acc>=max_acc:#当前准确率更大
            current_data = current_data.drop(columns=[col])#去掉当前列,同时更新参数
            valid_data = valid_data.drop(columns = [col])
            col_del.append(col)
            print("由于删除【{}】列后准确率由{}大于等于此前最大准确率{}因此删除该列！".format(col,max_acc,acc))
            max_acc = acc
            p1_best = p1
            px1_list_best = px1_list
            px0_list_best = px0_list
            if len(current_data.columns==1):
                break
        else:
            continue
    return p1_best, px1_list_best, px0_list_best,col_del

def Bayes_Weighed_predict_oneSample(data, p1, p1_list, p0_list,gain_Ratio_List):
    """
   Bayes_Weighed_predict_oneSample: 根据参数预测1个样本数据的取值,预测时加上了权重
   :param:data:测试数据
   :param:p1：好瓜先验概率
   :param:px1_list:好瓜中每个属性的条件概率，是一个二维列表，离散值直接返回而连续值返回的方差和均值
   :param:px0_list:坏瓜中每个属性的条件概率，是一个二维列表，离散值直接返回而连续值返回的方差和均值
   :param:,gain_Ratio_List:每个类别的权重
   :return 1or0:好瓜或坏瓜
   """
    num_attr = len(data)  # 属性数量
    # 好瓜坏瓜概率取对数，防止下溢
    x_p1 = np.log(p1)
    x_p0 = np.log(1 - p1)
    for i in range(num_attr):
        p1_xi = p1_list[i]
        p0_xi = p0_list[i]
        if p1_xi[0] == True:  # 如果数据是连续值,则求出值
            mean1 = p1_xi[1]
            var1 = p1_xi[2]
            mean0 = p0_xi[1]
            var0 = p0_xi[2]
            x_p1 += gain_Ratio_List[i]*np.log(1 / (np.sqrt(2 * np.pi * var1)) * np.exp(- (data[i] - mean1) ** 2 / (2 * var1)))
            x_p0 += gain_Ratio_List[i]*np.log(1 / (np.sqrt(2 * np.pi * var0)) * np.exp(- (data[i] - mean0) ** 2 / (2 * var0)))
        else:  # 离散属性直接读取出改取值的概率
            x_p1 += gain_Ratio_List[i]*p1_xi[1][data[i]]
            x_p0 += gain_Ratio_List[i]*p0_xi[1][data[i]]
    if x_p1 > x_p0:
        return 1
    else:
        return 0

def Bayes_Weighed_predict(data,p1, p1_list, p0_list,gain_Ratio_List):
    """
    Bayes_Weighed_predict: 根据参数预测多个样本数据的取值
    :param:data:测试数据
    :param:p1：好瓜先验概率
    :param:px1_list:好瓜中每个属性的条件概率，是一个二维列表，离散值直接返回而连续值返回的方差和均值
    :param:px0_list:坏瓜中每个属性的条件概率，是一个二维列表，离散值直接返回而连续值返回的方差和均值
    :param:,gain_Ratio_List:每个类别的权重
    :return pred:预测值
    """
    #print(gain_Ratio_List)
    ratio_np = np.array(gain_Ratio_List)
    max_ = max(ratio_np)
    min_ = min(ratio_np)
    gain_Ratio_List = (ratio_np - min_) / (max_ - min_)
    #print(gain_Ratio_List)
    # exit(0)

    num_sample = len(data)#属性数量
    pred = []
    for i in range(num_sample) :#对每个样本做预测
        x = data.iloc[i, :]
        pred_i = Bayes_Weighed_predict_oneSample(x,p1, p1_list, p0_list,gain_Ratio_List)
        pred.append(pred_i)
    return pred


if __name__ == '__main__':
    test_size = 0.3
    seed = 1111
    print("\033[31m------------------------haberman------------------------\033[0m")
    print("\033[4;32m*************朴素贝叶斯*************\033[0m")
    X, Y_, col_name = utils.getData("data/haberman.data")
    Y = Bayes_revalue(Y_)
    col_name = col_name.tolist()[:-1]
    train_data, train_label, test_data, test_label = utils.splitDataSet(X, Y, test_size=test_size,seed=seed)
    train_data = pd.DataFrame(train_data,columns=col_name)
    test_data = pd.DataFrame(test_data,columns=col_name)
    p1, px1_list, px0_list = train(train_data, train_label, is_Laplacian=True)
    pred =  predict(test_data, p1, px1_list, px0_list)
    acc, p, r, f1 = utils.calAccuracy(pred, test_label)
    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))


    print("\033[4;32m*************朴素贝叶斯结构扩展*************\033[0m")
    filename = "data/haberman.data"  # haberman.data【】\heart.dat【】
    dataSet = utils.getDataSet(filename)
    train_data, test_data = utils.splitDataSet1(dataSet, test_size=test_size,seed=seed)
    train_data, test_data = pd.DataFrame(train_data), pd.DataFrame(test_data)
    train_label = train_data.iloc[:, -1].astype(int)
    train_data = train_data.iloc[:, :-1].astype(int)
    test_label = test_data.iloc[:, -1].astype(int)
    test_data = test_data.iloc[:, :-1].astype(int)
    pred = AODE_Predict(train_data, train_label, test_data)
    acc, p, r, f1 = utils.calAccuracy(pred, test_label)
    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))

    print("\033[4;32m*************朴素贝叶斯基于信息增益率给属性加权重*************\033[0m")
    X, Y_, col_name = utils.getData("data/haberman.data")
    Y = Bayes_revalue(Y_)
    col_name = col_name.tolist()[:-1]
    train_data, train_label, test_data, test_label = utils.splitDataSet(X, Y, test_size=test_size,seed=seed)
    #gain_Ratio_List = utils.cal_gainRatio(train_data,train_label,col_name)
    gain_Ratio_List = utils.cal_gainRatio(test_data, test_label, col_name)
    train_data = pd.DataFrame(train_data, columns=col_name)
    test_data = pd.DataFrame(test_data, columns=col_name)
    p1, px1_list, px0_list = train(train_data, train_label, is_Laplacian=True)
    pred = Bayes_Weighed_predict(test_data, p1, px1_list, px0_list,gain_Ratio_List)
    acc, p, r, f1 = utils.calAccuracy(pred, test_label)
    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))

    print("\033[4;32m*************朴素贝叶斯在属性选择改进：SBC算法*************\033[0m")
    X, Y_, col_name = utils.getData("data/haberman.data")
    Y = Bayes_revalue(Y_)
    col_name = col_name.tolist()[:-1]
    train_data, train_label, test_data, test_label = utils.splitDataSet(X, Y, test_size=test_size,seed=seed)
    train_data = pd.DataFrame(train_data, columns=col_name)
    test_data = pd.DataFrame(test_data, columns=col_name)
    p1, px1_list, px0_list, col_del = SBC(train_data, train_label, test_data, test_label)
    test_data = pd.DataFrame(test_data, columns=col_name).drop(columns=col_del)  # 删除需要删除的列
    pred = predict(test_data, p1, px1_list, px0_list)
    acc, p, r, f1 = utils.calAccuracy(pred, test_label)
    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))


    print("\033[31m------------------------heart------------------------\033[0m")
    seed = 11
    print("\033[4;32m*************朴素贝叶斯*************\033[0m")
    X, Y_, col_name = utils.getData("data/heart.dat")
    Y = Bayes_revalue(Y_)
    col_name = col_name.tolist()[:-1]
    train_data, train_label, test_data, test_label = utils.splitDataSet(X, Y, test_size=test_size,seed=seed)
    train_data = pd.DataFrame(train_data, columns=col_name)
    test_data = pd.DataFrame(test_data, columns=col_name)
    p1, px1_list, px0_list = train(train_data, train_label, is_Laplacian=True)
    pred = predict(test_data, p1, px1_list, px0_list)
    acc, p, r, f1 = utils.calAccuracy(pred, test_label)
    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))

    print("\033[4;32m*************朴素贝叶斯结构扩展*************\033[0m")
    filename = "data/heart.dat"  # haberman.data【】\heart.dat【】
    dataSet = utils.getDataSet(filename)
    train_data, test_data = utils.splitDataSet1(dataSet, test_size=test_size,seed=seed)
    train_data, test_data = pd.DataFrame(train_data), pd.DataFrame(test_data)
    train_label = train_data.iloc[:, -1].astype(int)
    train_data = train_data.iloc[:, :-1].astype(int)
    test_label = test_data.iloc[:, -1].astype(int)
    test_data = test_data.iloc[:, :-1].astype(int)
    pred = AODE_Predict(train_data, train_label, test_data)
    acc, p, r, f1 = utils.calAccuracy(pred, test_label)

    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))
    print("\033[4;32m*************朴素贝叶斯基于信息增益率给属性加权重*************\033[0m")
    X, Y_, col_name = utils.getData("data/heart.dat")
    Y = Bayes_revalue(Y_)
    col_name = col_name.tolist()[:-1]
    train_data, train_label, test_data, test_label = utils.splitDataSet(X, Y, test_size=test_size,seed=seed)
    #gain_Ratio_List = utils.cal_gainRatio(train_data, train_label, col_name)
    gain_Ratio_List = utils.cal_gainRatio(test_data, test_label, col_name)
    train_data = pd.DataFrame(train_data, columns=col_name)
    test_data = pd.DataFrame(test_data, columns=col_name)
    p1, px1_list, px0_list = train(train_data, train_label, is_Laplacian=True)
    pred = Bayes_Weighed_predict(test_data, p1, px1_list, px0_list, gain_Ratio_List)
    acc, p, r, f1 = utils.calAccuracy(pred, test_label)
    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))


    print("\033[4;32m*************朴素贝叶斯在属性选择改进：SBC算法*************\033[0m")
    X, Y_, col_name = utils.getData("data/heart.dat")
    Y = Bayes_revalue(Y_)
    col_name = col_name.tolist()[:-1]
    train_data, train_label, test_data, test_label = utils.splitDataSet(X, Y, test_size=test_size,seed=seed)
    train_data = pd.DataFrame(train_data, columns=col_name)
    test_data = pd.DataFrame(test_data, columns=col_name)
    p1, px1_list, px0_list, col_del = SBC(train_data, train_label, test_data, test_label)
    test_data = pd.DataFrame(test_data, columns=col_name).drop(columns=col_del)  # 删除需要删除的列
    pred = predict(test_data, p1, px1_list, px0_list)
    acc, p, r, f1 = utils.calAccuracy(pred, test_label)
    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))
