'''对西瓜数据集3.0，编程实现一种多变量决策树算法，并进行可视化展示。'''
from collections import Counter
import numpy as np
import pandas as pd
from CH4_Decision_Tree import plotDecisionTree
from CH4_Decision_Tree.Task_4_3_DecisionTreeOnEnt import Node,get_dataset

def get_dataset2(filename='table4.3.txt'):
    """
      get_dataset2：从文件中读取数据，将其转化为指定格式的dataset和col_name
      和之前区别在于这个将离散属性给one-hot编码
      :returns  dataset:数据+标签 col_name:列名，即有哪些属性
      """
    df = pd.read_csv(filename)
    # 类别变量转化为数字变量
    # 色泽
    color = pd.get_dummies(df.色泽, prefix="色泽")
    # 根蒂
    root = pd.get_dummies(df.根蒂, prefix="根蒂")
    # 敲声
    knocks = pd.get_dummies(df.敲声, prefix="敲声")
    # 纹理
    texture = pd.get_dummies(df.纹理, prefix="纹理")
    # 脐部
    navel = pd.get_dummies(df.脐部, prefix="脐部")
    # 触感
    touch = pd.get_dummies(df.触感, prefix="触感")
    # 密度和含糖量
    densityAndsugar = df[['密度', '含糖率']]
    # 标签
    labels = df['好瓜']
    # 融合
    dataset = pd.concat([color, root, knocks, texture, navel, touch, densityAndsugar, labels], axis=1)
    col_name = dataset.columns.tolist()[:-1]  # 去除标签
    dataset = dataset.values.tolist()
    return dataset,col_name

class DataSet(object):
    """
    数据集类
    """
    def __init__(self, data_set, col_name):
        """
        :param col_name 列名 list[]
        :param data_set 数据集 list[list[]]:
        """
        self.num = len(data_set)  # 获得数据集大小
        self.data_set = data_set
        self.data = [example[:-1] for example in data_set if self.data_set[0]]  # 提取数据剔除标签
        self.labels = [int(example[-1]) for example in data_set if self.data_set[0]]  # 类别向量集合
        self.col_name = col_name  # 特征向量对应的标签

    def get_mostLabel(self):
        """
        得到最多的类别
        """
        class_dict = Counter(self.labels)
        return max(class_dict)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def logistic_test(self):
        train_data = np.array(self.data)
        w = np.ones((train_data.shape[1] + 1, 1))  # 随机初始化参数w，w shape[属性+1,1]
        add_col = np.ones((train_data.shape[0], 1))
        x = np.c_[train_data, add_col]  # 将原始数据增加一列即是x = (个数,属性+1)
        iteration = 0  # 记录训练次数
        lr = 0.01
        best_w = w
        max_correct = 0#预测最多数
        while iteration < 200:
            correct = 0
            pred = DataSet.sigmoid(np.dot(x, w))  # 计算预测值，预测值要用sigmoid函数，使其靠近0或者1
            # 梯度下降
            target = np.array(self.labels).reshape(self.num,1)
            b = pred - target   # 实际和预估的差值
            x_transpose = np.transpose(x)  # 矩阵转置,转置后[属性个数+1，样本个数]
            change = np.dot(x_transpose, b)
            w -= change * lr  # 更新权值
            iteration += 1
            for i in range(len(pred)):
                if (pred[i][0] <= 0.5 and self.labels[i]==0) or (pred[i][0] > 0.5 and self.labels[i]==1):
                   correct += 1
            if correct>max_correct:
                best_w = w
        return best_w

    def is_sameFeature(self):
        """
        判断数据集的所有特征是否相同
        :return: Ture or False
        """
        example = self.data[0]
        for index in range(self.num):
            if self.data[index] != example:
                return False
        return True

    def is_OneClass(self):
        """
        判断当前的data_set是否属于同一类别，如果是返回类别，如果不是，返回-1
        :return: class or -1
        """

        if len(np.unique(self.labels)) == 1:  # 只有一个类别
            return self.labels[0]
        else:
            return -1

    def split_data_set(self, w):
        """
        划分连续型数据
        :param weights:划分的权重
        :return:划分后的数据集
        """
        left_feature_list = []
        right_feature_list = []

        train_data = np.array(self.data)
        add_col = np.ones((train_data.shape[0], 1))
        x = np.c_[train_data, add_col]
        pred = DataSet.sigmoid(np.dot(x, w))
        for i in range(len(pred)):
            if pred[i][0] <= 0.5:
                left_feature_list.append(self.data_set[i])
            else:
                right_feature_list.append(self.data_set[i])
        return left_feature_list, right_feature_list

    def choose_best_weights(self):
        """
        根据现有数据集划分出信息增益的集合，返回划分属性的索引
        :return: 最佳权重，最佳划分属性索引
        """
        new_weights = self.logistic_test()
        best_col_name = str(new_weights[0])+'*'+self.col_name[0]
        for i in range(1,len(self.col_name)):
            best_col_name += '+' +str(new_weights[i]) + '*'+self.col_name[i]
            if i%2==1:
                best_col_name += '\n'
        best_col_name += '\n<='+str(-new_weights[-1])
        return new_weights, best_col_name

class DecisionTree(object):
    """
    决策树
    """

    def __init__(self, data_set: DataSet):
        self.root = self.bulidTree(data_set,(0,0))  # 头结点

    def bulidTree(self, data_set: DataSet,size):
        """
        :param self
        :param data_set: 数据集
        :return:
        """
        node = Node()
        if len(data_set.labels) == 0 or data_set.is_sameFeature():  # 如果数据集为空或者数据集的特征向量相同
            node.isLeaf = True   # 标记为叶子节点
            node.label = data_set.get_mostLabel()   # 标记叶子节点的类别为数据集中样本数最多的类
            return node
        if data_set.is_OneClass() != -1:  # 如果数据集中样本属于同一类别C
            node.isLeaf = True   # 标记为叶子节点
            node.label = data_set.is_OneClass()   # 标记叶子节点的类别为C类
            return node

        best_weights, best_label = data_set.choose_best_weights()  # 最佳划分数据集的标签索引,最佳划分数据集的标签
        node.attr = best_label   # 设置非叶节点的属性为最佳划分数据集的标签
        left_data_set, right_data_set = data_set.split_data_set(best_weights)
        '''这一步是防止无限递归'''
        if size in (len(left_data_set), len(right_data_set)):
            node.isLeaf = True  # 标记为叶子节点
            node.label = data_set.get_mostLabel()  # 标记叶子节点的类别为数据集中样本数最多的类
            return node
        left_node = Node()
        right_node = Node()
        if len(left_data_set) == 0:  # 如果子数据集为空
            left_node.isLeaf = True  # 标记新节点为叶子节点
            left_node.label = data_set.get_mostLabel()  # 类别为父数据集中样本数最多的类
        else:
            left_node = self.bulidTree(DataSet(left_data_set,data_set.col_name),(len(left_data_set), len(right_data_set)))
        if len(right_data_set) == 0:  # 如果子数据集为空
            right_node.isLeaf = True  # 标记新节点为叶子节点)
            right_node.label = data_set.get_mostLabel()  # 类别为父数据集中样本数最多的类
        else:
            right_node = self.bulidTree(DataSet(right_data_set,data_set.col_name),(len(left_data_set), len(right_data_set)))
        left_node.value = '小于等于'
        right_node.value = '大于'
        node.children.append(left_node)
        node.children.append(right_node)
        return node

dataset,col_name = get_dataset2('table4.3.txt')
Data_set = DataSet(dataset,col_name)
decisionTree = DecisionTree(Data_set)
plotDecisionTree.createPlot(decisionTree)













