# coding: utf-8
from numpy import *
import numpy as np
import pandas as pd
from collections import Counter
import CH4_Decision_Tree.plotDecisionTree
from CH4_Decision_Tree import plotDecisionTree


def get_dataset(filename ="table4.2.txt"):
    """
    get_dataset：从文件中读取数据，将其转化为指定格式的dataset和col_name
    :returns  dataset:训练集数据+标签 datatest:测试集数据+标签
    """
    df = pd.read_csv(filename)
    #下面分别提取标签和数据:一行对应一个数据
    data = df.values[:10, :].tolist()
    data_test = df.values[10:, :].tolist()
    col_name = df.columns.values[0:-1].tolist()#剔除最后一列
    #构建训练集以及测试集
    dataset = DataSet(data, col_name)
    datatest = DataSet(data_test, col_name)
    return dataset,datatest


class Node(object):
    """
    决策树节点类
    """
    def __init__(self):
        self.attr = None     # 决策节点属性
        self.value = None    # 节点与父节点的连接属性
        self.isLeaf = False  # 是否为叶节点 T or F
        self.label = 0    # 叶节点分类值 int
        self.parent = None   # 父节点
        self.children = []   # 子列表

    @staticmethod
    def set_leaf(self, label):
        """
        :param self:
        :param label: 叶子节点分类值 int
        :return:
        """
        self.isLeaf = True
        self.label = label


class DataSet(object):
    """
    数据集类
    """
    def __init__(self, data_set, col_name):
        """
        :param col_name 标签集合 list[]
        :param data_set 数据集 list[list[]]:
        """
        self.num = len(data_set)  # 获得数据中的向量个数
        self.data_set = data_set
        self.data = [example[:-1] for example in data_set if self.data_set[0]]  # 特征向量集合
        self.labels = [int(example[-1]) for example in data_set if self.data_set[0]]  # 类别向量集合
        self.col_name = col_name  # 特征向量对应的标签

    def get_mostLabel(self):
        """
        得到最多的类别
        """
        class_dict = Counter(self.labels)
        return max(class_dict)

    def is_OneClass(self):
        """
        判断当前的data_set是否属于同一类别，如果是返回类别，如果不是，返回-1
        :return: class or -1
        """
        if len(np.unique(self.labels))==1:#只有一个类别
            return self.labels[0]
        else:
            return -1

    def group_feature(self, index, feature):
        """
        group_feature:将 data[index] 等于 feature的值放在一起,同时去除该列，保证不会重复计算
        :param index:当前处理特征的索引
        :param feature:处理的特征值
        :return:聚集后的数据经过打包返回
        """
        grouped_data_set = [example[:index]+example[index+1:] for example in self.data_set if example[index] == feature]
        sub_col = self.col_name.copy()  # list 传参可能改变原参数值，故需要复制出新的labels避免影响原参数
        del(sub_col[index])       #删除划分属性
        return DataSet(grouped_data_set, sub_col)


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

    def cal_gini(self):
        """
        计算基尼指数
        :return:float 基尼指数
        """
        class_dict = {}
        for label in self.labels:
            if label not in class_dict.keys():
                class_dict[label] = 0
            class_dict[label] += 1
        # 计算基尼指数
        Gini = 1.0
        for key in class_dict.keys():
            prob = float(class_dict[key]) / self.num
            Gini -= prob * prob
        return Gini

    def choose_byGini(self):
        """
        根据现有数据集划分出基尼指数最小的集合，返回划分属性的索引
        :return: 最佳划分属性索引，int
        """
        num_features = len(self.data[0])
        best_gini = 1000
        best_feature_index = -1
        for index in range(num_features):#遍历每一个属性特征
            unique_feat = set([example[index] for example in self.data])
            new_gini = 0
            for feature in unique_feat:#计算每一个属性特征的基尼指数Gini
                sub_data_set = self.group_feature(index,feature)
                prob = sub_data_set.num/float(self.num)
                new_gini += prob*sub_data_set.cal_gini()
            if new_gini < best_gini:#记录最低基尼指数的值以及索引
                best_gini = new_gini
                best_feature_index = index
        return best_feature_index


class DecisionTree(object):
    """
    决策树
    """
    def __init__(self, data_set, data_test, feature_dict):
        self.root = self.bulidTree(data_set, data_test, feature_dict)
        self.data_set = data_set
        self.data_test = data_test

    @staticmethod
    def classify(node, vector_test, col_name):
        """
        :param node: 决策树根节点
        :param vector_test: 测试样例向量
        :param col_name: 特征属性名称
        :return:
        """
        label = node.attr
        feat_index = col_name.index(label)
        for child in node.children:
            if child.value == vector_test[feat_index]:
                if not child.isLeaf:
                    class_label = DecisionTree.classify(child,vector_test,col_name)
                else:
                    class_label = child.label
        return class_label

    @staticmethod
    def pre_test(data_set, data_test,best_index, feat_set):
        error = 0
        for feat in feat_set:
            sub_data_set = data_set.group_feature(best_index, feat)  # 划分数据集并返回子数据集
            sub_data_test = data_test.group_feature(best_index, feat)
            # print sub_data_set.get_mostLabel()
            # print sub_data_test.labels
            error += DecisionTree.test_majority(sub_data_set.get_mostLabel(), sub_data_test)
        print(error)
        return error


    def post_test(self, node, data_test):

        error = 0
        for i in range(data_test.num):
            if DecisionTree.classify(node, data_test.data[i], data_test.col_name) != data_test.labels[i]:
                error += 1
        return error

    @staticmethod
    def test_majority(majority, data_test):
        error = 0
        for i in range(data_test.num):
            if majority != data_test.labels[i]:
                error += 1
        return error


    def bulidTree(self, data_set, data_test, feature_dict):
        """
        :param self
        :param data_set: 数据集
        :return:
        """
        node = Node()
        if data_set.is_OneClass() != -1:  # 如果数据集中样本属于同一类别C
            node.isLeaf = True   # 标记为叶子节点
            node.label = data_set.is_OneClass()   # 标记叶子节点的类别为C类
            return node
        if len(data_set.labels) == 0 or data_set.is_sameFeature():  # 如果数据集为空或者数据集的特征向量相同
            node.isLeaf = True   # 标记为叶子节点
            node.label = data_set.get_mostLabel()   # 标记叶子节点的类别为数据集中样本数最多的类
            return node
        best_feature_index = data_set.choose_byGini()  # 最佳划分数据集的标签索引
        best_label = data_set.col_name[best_feature_index]   # 最佳划分数据集的标签
        node.attr = best_label   # 设置非叶节点的属性为最佳划分数据集的标签
        feat_set_full = feature_dict[best_label]
        feat_set = set([example[best_feature_index] for example in data_set.data])  # 最佳划分标签的可取值集合
        print(feat_set)
        # print DecisionTree.pre_test(data_set,data_test,best_feature_index,feat_set),DecisionTree.test_majority(data_set.get_mostLabel(),data_test)
        if DecisionTree.pre_test(data_set,data_test,best_feature_index,feat_set) >= \
                DecisionTree.test_majority(data_set.get_mostLabel(),data_test):   #当划分后的错误率 大于或者等于 不划分的错误率 不划分
            node.isLeaf = True  # 标记新节点为叶子节点
            node.label = data_set.get_mostLabel()  # 类别为父数据集中样本数最多的类
            return node

        for feat in feat_set:
            new_node = Node()
            sub_data_set = data_set.group_feature(best_feature_index, feat)   # 划分数据集并返回子数据集
            if sub_data_set.num == 0:   # 如果子数据集为空
                new_node.isLeaf = True  # 标记新节点为叶子节点
                new_node.label = data_set.get_mostLabel()   # 类别为父数据集中样本数最多的类
            else:
                new_node = self.bulidTree(sub_data_set,data_test.group_feature(best_feature_index, feat), feature_dict)
            new_node.value = feat  # 设置节点与父节点间的连接属性
            new_node.parent = node  # 设置父节点
            node.children.append(new_node)
        for feat in feat_set_full-feat_set:
            new_node = Node()
            new_node.isLeaf = True  # 标记新节点为叶子节点
            new_node.label = data_set.get_mostLabel()  # 类别为父数据集中样本数最多的类
            new_node.value = feat  # 设置节点与父节点间的连接属性
            new_node.parent = node  # 设置父节点
            node.children.append(new_node)
        return node


#生成特征属性字典
def generate_full_features(data_set):
    features_list = {}#初始化特征字典
    col_name = data_set.col_name#获取数据集的特征属性名称
    for i in range(len(data_set.data[0])):#遍历每一个属性
        new_feature = []
        for feature in data_set.data:
            new_feature.append(feature[i])#将每一个样例的对应属性值添加到new_feature列表中去
        features_list[col_name[i]] = set(new_feature)#将属性特征以及对应属性值添加到字典中去
    return features_list


data_set, data_test=get_dataset("table4.2.txt")#读取数据，划分数据集（训练集&测试集）
feature_dict = generate_full_features(data_set)#获取训练集的特征属性字典
decisionTree = DecisionTree(data_set, data_test,feature_dict)#构建决策树
plotDecisionTree.createPlot(decisionTree)#可视化决策树











