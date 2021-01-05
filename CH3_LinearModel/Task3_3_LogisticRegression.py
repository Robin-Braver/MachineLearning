"""编程实现对率回归，分析西瓜数据集3.0α上的运行结果。"""
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Sigmoid函数：将输入值转化为接近0或1的值
    :param x:需要处理的变量（这里多为矩阵）
    :return 经sigmoid函数处理后的值
    """
    return 1.0 / (1 + np.exp(-x))

def getData():
    """
    getData函数：从文件中读取数据，转化为指定的ndarray格式
    :returns  X：样本，格式为：每行一条数据，列表示属性
    :returns  label：对应的标签
    """
    label = []#标签
    density = []#密度
    sugar_rate = []#含糖量
    with open("西瓜数据集.txt",encoding='utf8') as fp:
        file_read = fp.readlines()#读取到的文件内容，一行为一个字符串存储在列表中
    for i in file_read:#由于读取到字符串，现对字符串处理，提取属性以及标签。
        tmp = i.split()#将每一行内容切割
        density.append(eval(tmp[1]))#提取密度信息到列表
        sugar_rate.append(eval(tmp[2]))#提取含糖量信息到列表
        label.append(eval(tmp[3]))#提取瓜的标签信息
    '''转化为np格式便于后续处理'''
    sugar_rate = np.array(sugar_rate)
    density = np.array(density)
    label = np.array(label).reshape(len(label),1)
    X = np.vstack((density, sugar_rate)).T#转换成每列代表属性，行代表样本
    return X,label

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
    '''分别计算TP FP FN TN'''
    for i in range(num):
        if pred[i, 0] > 0.5 and label[i]==1:
            TP += 1
        elif pred[i, 0] > 0.5 and label[i]==0:
            FP += 1
        elif pred[i, 0] <= 0.5 and label[i]==1:
            FN += 1
        elif pred[i, 0] <= 0.5 and label[i]==0:
            TN += 1
    '''计算准确率 查准率 查全率 F1度量'''
    acc = (TN+TP) / num
    if TP+FP==0:
        p = 0
    else:
        p = TP/(TP+FP)
    if TP+FN == 0:
        r = 0
    else:
        r = TP/(TP+FN)
    if p+r==0:
        f1 = 0
    else:
        f1 = 2*p*r/(p+r)
    return acc,p,r,f1

def train(train_data,train_label,epoch,lr):
    """
   train函数：训练逻辑斯蒂回归模型
   :param train_data:训练样本
   :param train_label:训练标签
   :param epoch:训练次数
   :param lr:学习率
   :return w:训练最终得到的参数
   """
    w = np.ones((train_data.shape[1]+1,1))#随机初始化参数w，w shape[属性+1,1]
    add_col = np.ones((train_data.shape[0],1))
    x = np.c_[train_data,add_col]#将原始数据增加一列即是x = (个数,属性+1)
    iteration = 0#记录训练次数
    Loss = []#记录损失值，用来画图
    P = []#记录查准率，用来画图
    R = []#记录查全率，用来画图
    while iteration<epoch:
        pred = sigmoid(np.dot(x,w))#计算预测值，预测值要用sigmoid函数，使其靠近0或者1
        #梯度下降
        b = pred - train_label #实际和预估的差值
        loss = np.average(np.abs(b))
        x_transpose = np.transpose(x)#矩阵转置,转置后[属性个数+1，样本个数]
        change = np.dot(x_transpose,b)
        w -= change*lr#更新权值
        iteration += 1
        acc,p,r,f1 = calAccuracy(pred,train_label)#计算正确率
        if iteration%10 == 0:
            Loss.append(loss)
            P.append(p)
            R.append(r)
        print("训练进度：{}/{}\tloss值：{:.6f}\t准确率：{:.4f}\t查准率：{:.4f}\t查全率：{:.4f}\tF1度量：{:.4f}"
              .format(iteration,epoch,loss,acc,p,r,f1))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    '''画损失函数图'''
    plt.figure()
    x_plt = [i for i in range(1, epoch//10 + 1)]
    plt.plot(x_plt,Loss)
    plt.title("训练损失变化图")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()
    '''画P-R曲线图'''
    plt.figure()
    plt.plot(R, P)
    plt.title("P-R曲线图")
    plt.ylabel('查准率P')
    plt.xlabel('查全率R')
    plt.show()
    return w

def test(test_data,test_label,w):
    """
      train函数：训练逻辑回归模型
      :param test_data:测试样本
      :param test_label:测试标签
      :param w:训练最终得到的参数
      """
    add_col = np.ones((test_data.shape[0], 1))
    x = np.c_[test_data, add_col]  # 将原始数据增加一列即是x = (个数,属性+1)
    pred = sigmoid(np.dot(x, w))  # 计算预测值，预测值要是用sigmoid函数，使其靠近0或者1
    acc,p,r,f1 = calAccuracy(pred, test_label)  # 计算正确率,召回率，查全率，F1度量
    b = pred - test_label  # 实际和预估的差值
    loss = np.average(np.abs(b))
    print("loss值：{:.6f}\t准确率：{:.4f}\t查准率：{:.4f}\t查全率：{:.4f}\tF1度量：{:.4f}".format(loss,acc,p,r,f1))

if __name__ == '__main__':
    data,label = getData()
    x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.3)#按照训练：测试=7:3比例划分训练集和测试集
    w = train(x_train,y_train,500,0.01)
    test(x_test,y_test,w)

    ''''sklearn中模型'''
    print()
    print("------------sklearn-----------")
    from sklearn import linear_model  # 表示，可以调用sklearn中的linear_model模块进行线性回归。
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc,p,r,f1 = calAccuracy(pred,y_test)
    print("准确率：{:.4f}\t查准率：{:.4f}\t查全率：{:.4f}\tF1度量：{:.4f}".format(acc,p,r,f1))

    '''对比中可以得出，其实模型很不稳定 高低起伏严重'''