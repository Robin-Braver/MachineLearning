'''编程实现标准BP算法，在西瓜数据集3.0上训练一个单隐层网络。通过调整迭代次数和学习率，分析代码收敛性。'''
import numpy as np
import matplotlib.pyplot as plt
import random

def sigmoid(x):
    """
    Sigmoid函数：将输入值转化为接近0或1的值
    :param x:需要处理的变量（这里多为矩阵）
    :return 经sigmoid函数处理后的值
    """
    return 1.0 / (1 + np.exp(-x))

# 标准BP算法
def BP_NN(data, q=10, lr=0.1,err=0.01,Epoch=200):
    """
    BP_NN:标准反向传播算法
    :param data: 带有标签的数据集
    :param q:隐层神经元个数
    :param lr:学习率
    :param err:提前终止误差
    :param Epoch:固定训练轮次
    :returns:v、w:最终即权重
    """
    X = data[:, :-1]#剔除最后一列即标签列
    X = np.insert(X, [0], -1, axis=1) #为X矩阵左边插入列-1来计算计算阈值
    Y = np.array([data[:, -1],1-data[:, -1]]).transpose()#与后面输出两个类别对应
    d = len(data[0, :])-1#输入神经元个数即是输入的属性个数,-1是减去标签
    l = len(np.unique(Y))#输出神经元个数，即要分几类
    v = np.mat(np.random.random((d+1, q)))#输入到隐层参数v：d*q+q个阈值
    w = np.mat(np.random.random((q+1, l)))#隐层到输出参数w：q*l+l个
    num_sample = len(data)#样本数即是行数
    Loss = []
    Accu = []
    for epoch in range(Epoch):
        correct = 0
        loss = 0
        for i in range(num_sample):#由于标准bp需要每个样本更新参数，因此需要逐样本训练
            b_init = sigmoid(np.mat(X[i, :]) * v) # 计算输入-隐藏  输出1*q matrix
            b = np.insert(b_init.T, [0], -1, axis=0)#同样加一列，来计算阈值,输出(q+1)*1 matrix
            y_cal = np.array(sigmoid(b.T * w))  # 隐藏-输出 输出：1*l array
            g = y_cal*(1-y_cal)*(Y[i, :]-y_cal) #计算 gj  1*l array
            e = np.array(b_init)*(1-np.array(b_init))*np.array((w[1:,:] * np.mat(g).T).T)#计算e 1*q array，从1切片剔除阈值
            #计算梯度
            d_w = lr * b * np.mat(g)
            d_v = lr * np.mat(X[i, :]).T * np.mat(e)
            #梯度更新
            w += d_w
            v += d_v
            loss = 0.5*np.sum((Y[i, :] - y_cal) ** 2)#均方误差
            if y_cal[0,0]>=y_cal[0,1]:
                pred = 1
            else:
                pred = 0
            correct += (pred==Y[i][0])
        #loss = loss/num_sample

        Loss.append(loss)
        Accu.append(correct / len(data))
        print("训练进度{}/{}\t\t误差{:.6f}\t准确率：{}".format(epoch + 1, Epoch, loss, correct / len(data)))
        if loss <= err:
            print("在第{}轮,均方误差{:.6f}小于指定停止误差{},因此提前停止训练！".format(epoch + 1, loss, err))
            break
    plot(epoch,Loss,Accu)
    return v, w

def calAccuracy(pred,label):
    """
   calAccuracy函数：计算准确率
   :param pred:预测值,二分类是两个类别概率
   :param label:真实值
   """
    label = label.reshape(max(label.shape[0],label.shape[1]))
    num = len(pred)  # 样本个数
    pred_label = []#将概率转换为标签
    for p in pred:
        if p[0]>=p[1]:
            pred_label.append(1)
        else:
            pred_label.append(0)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    '''分别计算TP FP FN TN'''
    for i in range(num):
        if pred_label[i] == 1 and label[i] == 1:
            TP += 1
        elif pred_label[i] ==1 and label[i] == 0:
            FP += 1
        elif pred_label[i] ==0 and label[i] == 1:
            FN += 1
        elif pred_label[i] ==0 and label[i] == 0:
            TN += 1
    '''计算准确率 查准率 查全率 F1度量'''
    acc = (TN + TP) / num
    if TP + FP == 0:
        p = 0
    else:
        p = TP / (TP + FP)
    if TP + FN == 0:
        r = 0
    else:
        r = TP / (TP + FN)
    if p + r == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
    print("准确率：{}/{}={:.4f}\t查准率：{:.4f}\t查全率：{:.4f}\tF1度量：{:.4f}".format((TN + TP),num,acc, p, r, f1))

def plot(epoch,loss,acc):
    """
  plot函数：画图
  :param epoch:画图的轮数
  :param loss:损失
  :param acc:正确率
  """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    '''画损失函数图'''
    plt.figure()
    x_plt = [i for i in range(1, epoch + 2)]
    plt.plot(x_plt, loss)
    plt.title("训练损失变化图")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()
    '''画正确率图'''
    plt.figure()
    plt.plot(x_plt, acc)
    plt.title("训练正确率变化图")
    plt.ylabel('accuracy')
    plt.xlabel('Epochs')
    plt.show()


def test(data, v, w):
    """
    test函数：计算正确率等
    :param data:测试数据
    :param v:输入到隐层参数
    :param w:隐层到输出参数
    :return
    """
    X = data[:, :-1]
    X = np.insert(X, [0], -1, axis=1)
    Y = np.array([data[:, -1]])
    #y_cal = np.zeros((len(data), 2))
    b_init = sigmoid(np.mat(X) * v)
    b = np.insert(b_init.T, [0], -1, axis=0)
    pred = np.array(sigmoid(b.T * w))#输出是概率
    calAccuracy(pred,Y)

def splitDataSet(data,test_size=0.3):
    """
    splitDataSet函数：划分数据集
    :param data:待划分样本
    :param test_size:测试集占比
    :param seed:随机数种子
    :returns testdata:测试集 train_size:训练集
    """
    total_num = len(data)
    test_num = int(total_num*test_size)
    np.random.shuffle(data)
    test_data = data[0:test_num,]
    train_data = data[test_num:total_num,]
    return train_data,test_data

#数据预处理
'''对于离散属性值使用如下规则
色泽：青绿-1 乌黑-2 浅白-3   根蒂：蜷缩-1 稍蜷-2 硬挺-3   敲声：浊响-1 沉闷-2 清脆-3
纹理：清晰-1 稍糊-2 模糊-3   脐部：凹陷-1 稍凹-2 平坦-3   触感：硬滑-1 软粘-2   
'''
Data = np.array([
    [1, 1, 1, 1, 1, 1, 0.697, 0.460, 1],
    [2, 1, 2, 1, 1, 1, 0.774, 0.376, 1],
    [2, 1, 1, 1, 1, 1, 0.634, 0.264, 1],
    [1, 1, 2, 1, 1, 1, 0.608, 0.318, 1],
    [3, 1, 1, 1, 1, 1, 0.556, 0.215, 1],
    [1, 2, 1, 1, 2, 2, 0.403, 0.237, 1],
    [2, 2, 1, 2, 2, 2, 0.481, 0.149, 1],
    [2, 2, 1, 1, 2, 1, 0.437, 0.211, 1],
    [2, 2, 2, 2, 2, 1, 0.666, 0.091, 0],
    [1, 3, 3, 1, 3, 2, 0.243, 0.267, 0],
    [3, 3, 3, 3, 3, 1, 0.245, 0.057, 0],
    [3, 1, 1, 3, 3, 2, 0.343, 0.099, 0],
    [1, 2, 1, 2, 1, 1, 0.639, 0.161, 0],
    [3, 2, 2, 2, 1, 1, 0.657, 0.198, 0],
    [2, 2, 1, 1, 2, 2, 0.360, 0.370, 0],
    [3, 1, 1, 3, 3, 1, 0.593, 0.042, 0],
    [1, 1, 2, 2, 2, 1, 0.719, 0.103, 0]])

train_data,test_data = splitDataSet(Data,0.3)
v,w = BP_NN(data=train_data,q=10,lr=0.1,err=0.001,Epoch=1000)
test(test_data,v, w)

