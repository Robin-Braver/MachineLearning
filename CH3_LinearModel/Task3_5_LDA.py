"""编程实现线性判别分析，分析西瓜数据集3.0α上的运行结果。"""
import numpy as np
import matplotlib.pyplot as plt

def getData():
    """
    getData函数：从文件中读取数据，转化为指定的ndarray格式
    :returns  X0：标签为0的样本
    :returns  X1：标签为1的样本
    """
    X0 = []
    X1 = []
    with open("西瓜数据集.txt",encoding='utf8') as fp:
        file_read = fp.readlines()#读取到的文件内容，一行为一个字符串存储在列表中
    for i in file_read:#由于读取到字符串，现对字符串处理，提取属性以及标签。
        tmp = i.split()#将每一行内容切割
        tmp.pop(0)#这一步是除去索引
        tmp_label = eval(tmp.pop(-1))#提取标签
        tmp = [eval(i) for i in tmp]#将字符串转为数字
        if tmp_label==0:#是反例子
            X0.append(tmp)
        else:
            X1.append(tmp)
    '''转化为np格式便于后续处理'''
    X0 = np.array(X0)
    X1 = np.array(X1)
    return X0,X1


def LDA(x0,x1):
    """
    LDA函数：根据接收到两类样本，计算w
    :return  w
    """
    # 求均值
    u0 = np.mean(x0, axis=0)
    u1 = np.mean(x1, axis=0)
    #求协方差矩阵
    conv0 = np.dot((x0-u0).T,(x0-u0))
    conv1 = np.dot((x1 - u1).T, (x1 - u1))
    #计算类内散度矩阵
    Sw = conv0+conv1
    #由拉格朗日乘法子计算w
    w = np.dot(np.mat(Sw).I, (u0 - u1).reshape(len(u0), 1))
    return w

def plot(w,X0,X1):
    """
    plot函数：对结果进行可视化
    :param w
    :param X0,X1:两类样本
    """
    #画直线
    k = w[0]/w[1]
    x_line = np.arange(0, 1,0.01)
    yy = k[0][0] * x_line
    yy = np.ravel(yy)#降维

    #计算反例在直线上的投影点
    xi = []
    yi = []
    for i in range(0, len(X0)):
        y0 = X0[i, 1]
        x0 = X0[i, 0]
        x1 = (k * y0 + x0) / (k ** 2 + 1)
        y1 = k * x1
        xi.append(x1)
        yi.append(y1)
    xi = np.array(xi).reshape(len(xi),1)
    yi = np.array(yi).reshape(len(yi),1)
    # 计算第二类样本在直线上的投影点
    xj = []
    yj = []
    for i in range(0, len(X1)):
        y0 = X1[i, 1]
        x0 = X1[i, 0]
        x1 = (k * y0 + x0) / (k ** 2 + 1)
        y1 = k * x1
        xj.append(x1)
        yj.append(y1)
    xj = np.array(xj).reshape(len(xj), 1)
    yj = np.array(yj).reshape(len(yj), 1)

    plt.figure()
    plt.plot(x_line, yy)
    #绘制初始数据散点图
    plt.scatter(X0[:,0],X0[:,1], marker='+',color='b')
    plt.scatter(X1[:,0],X1[:,1], marker='*',color='r')
    # 画出投影后的点
    plt.plot(xi, yi, marker='.',color='b')
    plt.plot(xj, yj, marker='.',color='r')
    plt.show()

if __name__ == '__main__':
    X0,X1 = getData()
    w = LDA(X0,X1)
    plot(w,X0,X1)
   
