"""以西瓜数据集3.0a的密度为输入，含糖率为输出，试使用LIBSVM训练一个SVR"""
from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt

def getData():
    """
    getData函数：从文件中读取数据，转化为指定的ndarray格式
    :returns  input：输入：密度
    :returns  output：输出：含糖量
    """
    density = []#密度
    sugar_rate = []#含糖量
    with open("西瓜数据集.txt",encoding='utf8') as fp:
        file_read = fp.readlines()#读取到的文件内容，一行为一个字符串存储在列表中
    for i in file_read:#由于读取到字符串，现对字符串处理，提取属性以及标签。
        tmp = i.split()#将每一行内容切割
        density.append(eval(tmp[1]))#提取密度信息到列表
        sugar_rate.append(eval(tmp[2]))#提取含糖量信息到列表
    '''转化为np格式便于后续处理'''
    output = np.array(sugar_rate)
    input = np.array(density)
    return input,output

#准备数据
X,y=getData()#X：密度，y：含糖率
X_ = [[i] for i in X]#格式转换：ndarray->list
y_ = [i for i in y]#格式转换：ndarray->list


#准备模型
line = svm.SVR(kernel='linear',C=1)#线性核
rbf1 = svm.SVR(kernel='rbf', C=1,gamma=2000 )#高斯核
rbf2 = svm.SVR(kernel='rbf', C=0.1, gamma=2000)#高斯核，惩罚参数C=5
poly = svm.SVR(kernel='poly', C=1, degree=5, gamma='auto')#多项式核

#训练模型
line.fit(X_, y_)
rbf1.fit(X_, y_)
rbf2.fit(X_, y_)
poly.fit(X_, y_)

#预测数据
#线性核
loss0 = 0
loss1 = 0
loss2 = 0
loss3 = 0


result0 = line.predict(X_)
print("-------------线性核-------------")
print('实际结果', y)
print('预测结果',result0)

#高斯核，惩罚参数C=1
result1 = rbf1.predict(X_)
print("------------高斯核，惩罚参数C=1------------")
print('实际结果', y)
print('预测结果',result1)

#高斯核，惩罚参数C=20
result2 = rbf2.predict(X_)
print("------------高斯核，惩罚参数C=0.1------------")
print('实际结果', y)
print('预测结果',result2)

#多项式核
result3 = poly.predict(X_)
print("-------------多项式核-------------")
print('实际结果', y)
print('预测结果',result3)

for i in range(len(X_)):
    loss0 += abs(y[i]-result0[i])
    loss1 += abs(y[i] - result1[i])
    loss2 += abs(y[i] - result2[i])
    loss3 += abs(y[i] - result3[i])

print("他们的误差分别为：{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(loss0,loss1,loss2,loss3))

#数据可视化
plt.plot(X, y, 'bo', fillstyle='none',label ="原数据")
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.title("密度-含糖率SVR")
plt.xlabel("密度")
plt.ylabel("含糖率")
plt.plot(X, result0, 'b*',label ="线性核" )#线性核
plt.plot(X, result1, 'r.',label ="高斯核，惩罚参数C=1")#高斯核，惩罚参数C=1
plt.plot(X, result2, 'g.',label ="高斯核，惩罚参数C=0.1")#高斯核，惩罚参数C=20
plt.plot(X, result3, 'c+',label ="多项式核")#多项式核
plt.legend()
plt.show()
