'''UCI网站的Iris和Wine两个数据集，比较10折交叉验证法和留一法所估计的对率回归的错误率'''
from random import shuffle
from CH3_LinearModel.Task3_3_LogisticRegression import *

def getData(filename):
    """
    getData函数：从文件中读取数据，转化为指定的ndarray格式
    :param filename:读取数据的文件名
    :returns  X：样本，格式为：每行一条数据，列表示属性
    :returns  label：对应的标签
    """
    X = []
    label = []
    #由于两个数据集结构不一致，因此处理方式不同
    if filename=="wine.data":
        with open(filename) as fp:
            file_read = fp.readlines()#读取到的文件内容，一行为一个字符串存储在列表中
            file_read = file_read[0:130]#只取两个类别
            shuffle(file_read)#打乱，方便后续划分
        for i in file_read:#由于读取到字符串，现对字符串处理，提取属性以及标签。
            tmp = i.split(',')#将每一行内容切割
            tmp_label = eval(tmp.pop(0))-1#提取标签，并将内容从原始列表弹出,并转化为数字
            label.append([tmp_label])#将一行标签加入到标签列表
            tmp = [eval(i)for i in tmp]#将每个属性由字符转为数字
            X.append(tmp)#提取属性值
    elif filename=="iris.data":
        with open(filename) as fp:
            file_read = fp.readlines()  # 读取到的文件内容，一行为一个字符串存储在列表中
            file_read = file_read[0:100]#只取两个类别
            shuffle(file_read)
        for i in file_read:  # 由于读取到字符串，现对字符串处理，提取属性以及标签。
            tmp = i.split(',')  # 将每一行内容切割
            tmp_label = tmp.pop(-1) # 提取标签，并将内容从原始列表弹出
            tmp_label = tmp_label.strip()#这一步去除回车
            if tmp_label=="Iris-setosa":#将类别转化为数字
                tmp_label=0
            elif tmp_label=='Iris-versicolor':
                tmp_label=1
            else:
                tmp_label=2
            label.append([tmp_label])  # 将一行标签加入到标签列表
            tmp = [eval(i) for i in tmp]  # 将每个属性由字符转为数字
            X.append(tmp)  # 提取属性值
    else:
        print("无此文件！")
        return None
    '''转化为np格式便于后续处理'''
    X = np.array(X)
    label = np.array(label)
    return X, label

def getData1(filename):
    """
    getData函数：从文件中读取数据，转化为指定的ndarray格式
    :param filename:读取数据的文件名
    :returns  X：样本，格式为：每行一条数据，列表示属性
    :returns  label：对应的标签
    """
    X = []
    label = []
    #由于两个数据集结构不一致，因此处理方式不同
    if filename=="wine.data":
        with open(filename) as fp:
            file_read = fp.readlines()#读取到的文件内容，一行为一个字符串存储在列表中
            file_read = file_read[59:178]#只取两个类别
            shuffle(file_read)#打乱，方便后续划分
        for i in file_read:#由于读取到字符串，现对字符串处理，提取属性以及标签。
            tmp = i.split(',')#将每一行内容切割
            tmp_label = eval(tmp.pop(0))-2#提取标签，并将内容从原始列表弹出,并转化为数字
            label.append([tmp_label])#将一行标签加入到标签列表
            tmp = [eval(i)for i in tmp]#将每个属性由字符转为数字
            X.append(tmp)#提取属性值
    elif filename=="iris.data":
        with open(filename) as fp:
            file_read = fp.readlines()  # 读取到的文件内容，一行为一个字符串存储在列表中
            file_read = file_read[50:150]#只取两个类别
            shuffle(file_read)
        for i in file_read:  # 由于读取到字符串，现对字符串处理，提取属性以及标签。
            tmp = i.split(',')  # 将每一行内容切割
            tmp_label = tmp.pop(-1) # 提取标签，并将内容从原始列表弹出
            tmp_label = tmp_label.strip()#这一步去除回车
            if tmp_label=="Iris-setosa":#将类别转化为数字
                tmp_label=2
            elif tmp_label=='Iris-versicolor':
                tmp_label=0
            else:
                tmp_label=1
            label.append([tmp_label])  # 将一行标签加入到标签列表
            tmp = [eval(i) for i in tmp]  # 将每个属性由字符转为数字
            X.append(tmp)  # 提取属性值
    else:
        print("无此文件！")
        return None
    '''转化为np格式便于后续处理'''
    X = np.array(X)
    label = np.array(label)
    return X, label

def getData2(filename):
    """
    getData函数：从文件中读取数据，转化为指定的ndarray格式
    :param filename:读取数据的文件名
    :returns  X：样本，格式为：每行一条数据，列表示属性
    :returns  label：对应的标签
    """
    X = []
    label = []
    #由于两个数据集结构不一致，因此处理方式不同
    if filename=="wine.data":
        with open(filename) as fp:
            file_read = fp.readlines()#读取到的文件内容，一行为一个字符串存储在列表中
            file_read1 = file_read[130:178]#只取两个类别
            file_read2 = file_read[0:59]
            file_read=file_read1+file_read2
            shuffle(file_read)#打乱，方便后续划分
        for i in file_read:#由于读取到字符串，现对字符串处理，提取属性以及标签。
            tmp = i.split(',')#将每一行内容切割
            tmp_label = eval(tmp.pop(0))#提取标签，并将内容从原始列表弹出,并转化为数字
            if tmp_label==3:#将类别转化为数字
                tmp_label=0
            elif tmp_label==0:
                tmp_label=1
            label.append([tmp_label])#将一行标签加入到标签列表
            tmp = [eval(i)for i in tmp]#将每个属性由字符转为数字
            X.append(tmp)#提取属性值
    elif filename=="iris.data":
        with open(filename) as fp:
            file_read = fp.readlines()  # 读取到的文件内容，一行为一个字符串存储在列表中
            file_read1 = file_read[100:150]  # 只取两个类别
            file_read2 = file_read[0:50]
            file_read = file_read1 + file_read2
            shuffle(file_read)
        for i in file_read:  # 由于读取到字符串，现对字符串处理，提取属性以及标签。
            tmp = i.split(',')  # 将每一行内容切割
            tmp_label = tmp.pop(-1) # 提取标签，并将内容从原始列表弹出
            tmp_label = tmp_label.strip()#这一步去除回车
            if tmp_label=="Iris-setosa":#将类别转化为数字
                tmp_label=1
            elif tmp_label=='Iris-versicolor':
                tmp_label=2
            else:
                tmp_label=0
            label.append([tmp_label])  # 将一行标签加入到标签列表
            tmp = [eval(i) for i in tmp]  # 将每个属性由字符转为数字
            X.append(tmp)  # 提取属性值
    else:
        print("无此文件！")
        return None
    '''转化为np格式便于后续处理'''
    X = np.array(X)
    label = np.array(label)
    return X, label

def cross_validation(data,label,k=10):
    """
    cross_validation函数：使用交叉验证法对数据集进行划分
    :param data:数据
    :param label:标签
    :param k:划分互斥数据集个数
    :returns  X_split：划分后的样本，格式为：每行一条数据，列表示属性
    :returns  Label_split：划分后对应的标签
    """
    data_split = []
    label_split = []
    num_per_set = len(label)//k#每个小子集对应数目
    '''以下确定每组个数'''
    num = [num_per_set for i in range(k)]
    if np.sum(num)<len(label):
        for i in range(len(label)-np.sum(num)):
            num[i] += 1
    index = 0
    for i in num:
        tmp_data = data[index:index+i,:].tolist()
        tmp_label = label[index:index +i, :].tolist()
        data_split.append(tmp_data)
        label_split.append(tmp_label)
        index += i###################i#####################num_per_set
    X_split = np.array(data_split)
    Label_split = np.array(label_split)
    return X_split,Label_split

def fit(data,label,epoch,lr):
    """
      fit函数：训练逻辑斯蒂回归模型，并测试
      :param data:处理后的数据集，即划分了10个子集
      :param train_label:处理后的标签，即划分了10个子集
      :param epoch:训练次数
      :param lr:学习率
      :return w:训练最终得到的参数
      """
    Loss = []  # 记录损失值，用来画图
    Accu = [] #记录正确率

    for j in range(epoch):
        test_data = np.array(data[j])#j是测试样本
        test_label = label[j]
        w = np.ones((test_data.shape[1] + 1, 1))  # 随机初始化参数w，w shape[属性+1,1]
        """"训练"""
        for i in range(len(data)):
            if i==j:
                continue#这一次选到的是测试样本 直接下一次
            train_data = np.array(data[i])
            train_label = label[i]
            add_col = np.ones((train_data.shape[0], 1))
            x = np.c_[train_data, add_col]  # 将原始数据增加一列即是x = (个数,属性+1)
            pred = sigmoid(np.dot(x, w))  # 计算预测值，预测值要是用sigmoid函数，使其靠近0或者1
            # 梯度下降
            b = pred - train_label  # 实际和预估的差值
            x_transpose = np.transpose(x)  # 矩阵转置,转置后[属性个数+1，样本个数]
            change = np.dot(x_transpose, b)
            w -= change * lr  # 更新权值
        """"测试"""
        add_col = np.ones((test_data.shape[0], 1))
        x = np.c_[test_data, add_col]  # 将原始数据增加一列即是x = (个数,属性+1)
        pred = sigmoid(np.dot(x, w))  # 计算预测值，预测值要是用sigmoid函数，使其靠近0或者1
        acc, p, r, f1 = calAccuracy(pred, test_label)  # 计算正确率,召回率，查全率，F1度量
        b = pred - test_label  # 实际和预估的差值
        loss = np.average(np.abs(b))
        Loss.append(loss)
        Accu.append(acc)
        print("取第{}个数据集为测试集时：\tloss值：{:.6f}\t准确率：{:.4f}" .format(j+1, loss, acc))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    '''画损失函数图'''
    plt.figure()
    x_plt = [i for i in range(1,epoch+1)]
    plt.plot(x_plt, Loss)
    plt.title("训练损失变化图")
    plt.ylabel('Loss')
    plt.xlabel('数据集')
    plt.show()
    return

print("\n----iris.data----")
print("--------------------------1------------------------------")
filename = "iris.data"
data,label = getData(filename)
print("\n----10折交叉验证法----")
data1,label1 = cross_validation(data,label,k=10)
fit(data1,label1,10,0.01)
print("\n----留一法----")
data1,label1 = cross_validation(data,label,k=100)
fit(data1,label1,10,0.01)
print("--------------------------2------------------------------")
data,label = getData1(filename)
print("\n----10折交叉验证法----")
data1,label1 = cross_validation(data,label,k=10)
fit(data1,label1,10,0.01)
print("\n----留一法----")
data1,label1 = cross_validation(data,label,k=100)
fit(data1,label1,10,0.01)
print("--------------------------3------------------------------")
data,label = getData2(filename)
print("\n----10折交叉验证法----")
data1,label1 = cross_validation(data,label,k=10)
fit(data1,label1,10,0.01)
print("\n----留一法----")
data1,label1 = cross_validation(data,label,k=100)
fit(data1,label1,10,0.01)





print("\n----wine.data----")
print("--------------------------1------------------------------")
filename = "wine.data"
print("\n----10折交叉验证法----")
data,label = getData(filename)
data1,label1 = cross_validation(data,label,k=10)
fit(data1,label1,10,0.01)
print("\n----留一法----")
data1,label1 = cross_validation(data,label,k=130)
fit(data1,label1,10,0.01)
print("--------------------------2------------------------------")
data,label = getData1(filename)
print("\n----10折交叉验证法----")
data1,label1 = cross_validation(data,label,k=10)
fit(data1,label1,10,0.01)
print("\n----留一法----")
data1,label1 = cross_validation(data,label,k=100)
fit(data1,label1,10,0.01)
print("--------------------------3------------------------------")
data,label = getData2(filename)
print("\n----10折交叉验证法----")
data1,label1 = cross_validation(data,label,k=10)
fit(data1,label1,10,0.01)
print("\n----留一法----")
data1,label1 = cross_validation(data,label,k=100)
fit(data1,label1,10,0.01)