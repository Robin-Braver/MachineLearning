from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Node(object):
    def __init__(self):
        self.feature_index = None
        self.split_point = None
        self.deep = None
        self.left_tree = None
        self.right_tree = None
        self.leaf_class = None

def gini(y, D):
    '''
    计算样本集y下的加权基尼指数
    :param y: 数据样本标签
    :param D: 样本权重
    :return: 加权后的基尼指数
    '''
    unique_class = np.unique(y)
    total_weight = np.sum(D)
    gini = 1
    for c in unique_class:
        gini -= (np.sum(D[y == c]) / total_weight) ** 2
    return gini


def calcMinGiniIndex(a, y, D):
    '''
    计算特征a下样本集y的的基尼指数
    :param a: 单一特征值
    :param y: 数据样本标签
    :param D: 样本权重
    :return:
    '''
    feature = np.sort(a)
    total_weight = np.sum(D)
    split_points = [(feature[i] + feature[i + 1]) / 2 for i in range(feature.shape[0] - 1)]
    min_gini = float('inf')
    min_gini_point = None
    for i in split_points:
        yv1 = y[a <= i]
        yv2 = y[a > i]
        Dv1 = D[a <= i]
        Dv2 = D[a > i]
        gini_tmp = (np.sum(Dv1) * gini(yv1, Dv1) + np.sum(Dv2) * gini(yv2, Dv2)) / total_weight
        if gini_tmp < min_gini:
            min_gini = gini_tmp
            min_gini_point = i
    return min_gini, min_gini_point

def chooseFeatureToSplit(X, y, D):
    '''
    :param X:
    :param y:
    :param D:
    :return: 特征索引, 分割点
    '''
    gini0, split_point0 = calcMinGiniIndex(X[:, 0], y, D)
    gini1, split_point1 = calcMinGiniIndex(X[:, 1], y, D)
    if gini0 > gini1:
        return 1, split_point1
    else:
        return 0, split_point0

def createSingleTree(X, y, D, deep=1,max_depth = 2):
    '''
    这里以C4.5 作为基学习器，限定深度为2，使用基尼指数作为划分点，基尼指数的计算会基于样本权重，
    不确定这样的做法是否正确，但在西瓜书p87, 4.4节中, 处理缺失值时, 其计算信息增益的方式是将样本权重考虑在内的，
    这里就参考处理缺失值时的方法。
    :param X: 训练集特征
    :param y: 训练集标签
    :param D: 训练样本权重
    :param deep: 树的深度
    :param max_depth:树最大深度
    :return:
    '''
    node = Node()
    node.deep = deep
    if (deep == max_depth)|(X.shape[0] <= 1): # 当前分支下，样本数量只有1个 或者 深度达到最大深度时，直接设置为也节点
        pos_weight = np.sum(D[y == 1])
        neg_weight = np.sum(D[y == -1])
        if pos_weight >= neg_weight:
            node.leaf_class = 1
        else:
            node.leaf_class = -1
        return node
    elif(len(np.unique(y))==1):#属于一个类别
        node.leaf_class = y[0]
        return node
    elif(len(np.unique(X))==1):#所有样本属性取值相同
        node.leaf_class = max(Counter(y))
        return node
    feature_index, split_point = chooseFeatureToSplit(X, y, D)
    node.feature_index = feature_index
    node.split_point = split_point
    left = X[:, feature_index] <= split_point
    right = X[:, feature_index] > split_point
    node.left_tree = createSingleTree(X[left, :], y[left], D[left], deep + 1,max_depth)
    node.right_tree = createSingleTree(X[right, :], y[right], D[right], deep + 1,max_depth)
    return node


def predictSingle(tree, x):
    '''
    基于基学习器，预测单个样本
    :param tree:
    :param x:
    :return:
    '''
    if tree.leaf_class is not None:
        return tree.leaf_class
    if x[tree.feature_index] > tree.split_point:
        return predictSingle(tree.right_tree, x)
    else:
        return predictSingle(tree.left_tree, x)

def predictBase(tree, X):
    '''
    基于基学习器预测所有样本
    :param tree:
    :param X:
    :return:
    '''
    result = []
    for i in range(X.shape[0]):
        result.append(predictSingle(tree, X[i, :]))
    return np.array(result)

def adaBoostTrain(X, y, tree_num=3,max_depth=2):
    '''
    以决策树作为基学习器，训练adaBoost
    :param X:
    :param y:
    :param tree_num:多少颗树
    :param max_depth:树最大深度
    :return:
    '''
    D = np.ones(y.shape) / y.shape  # 初始化权重
    trees = []  # 所有基学习器
    a = []  # 基学习器对应权重
    for _ in range(tree_num):
        tree = createSingleTree(X, y, D,max_depth=max_depth)
        hx = predictBase(tree, X)
        err_rate = np.sum(D[hx != y])
        at = np.log((1 - err_rate) / max(err_rate, 1e-16)) / 2
        trees.append(tree)
        a.append(at)
        if (err_rate > 0.5)|(err_rate == 0): #错误率大于0.5或者错误率为0时则直接停止
            break
        # 更新每个样本权重
        err_index = np.ones(y.shape)
        err_index[hx == y] = -1
        D = D * np.exp(err_index * at)
        D = D / np.sum(D)
    return trees, a


def adaBoostPredict(X, trees, a):
    agg_est = np.zeros((X.shape[0],))
    for tree, am in zip(trees, a):
        agg_est += am * predictBase(tree, X)
    result = np.ones((X.shape[0],))
    result[agg_est < 0] = -1
    return result.astype(int)


def pltAdaBoostDecisionBound(X_, y_, trees, a):
    pred = adaBoostPredict(X_,trees,a)
    correct = 0
    for i in range(len(pred)):
        if pred[i] == y_[i]:
            correct += 1
    acc = "---正确率为："+str(correct)+'/'+str(len(pred))+'='+str(correct*1./len(pred))
    pos = y_ == 1
    neg = y_ == -1
    x_tmp = np.linspace(0, 0.8, 100)
    y_tmp = np.linspace(0, 0.6, 100)
    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)
    X_ravel = X_tmp.ravel()
    Y_ravel = Y_tmp.ravel()
    Z_ = adaBoostPredict(np.c_[X_ravel, Y_ravel], trees, a).reshape(X_tmp.shape)

    for tree in trees:#划分基分类器
        split_point = tree.split_point
        feature_index = tree.feature_index
        if feature_index==0:#以密度划分
            x_ = [split_point for i in x_tmp]
            plt.plot(x_,y_tmp, color='gray', linestyle=':')
        else:
            y_ = [split_point for i in x_tmp]
            plt.plot(x_tmp, y_, color='gray', linestyle=':')

    plt.contour(X_tmp, Y_tmp, Z_, [0],colors='red', linewidths=1)
    plt.scatter(X_[pos, 0], X_[pos, 1], marker='+', label='好瓜', color='k')
    plt.scatter(X_[neg, 0], X_[neg, 1], marker='_', label='坏瓜', color='k')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.title(str(len(trees))+"个基学习器"+acc)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('西瓜数据集3a.txt', sep=' ')
    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values
    y[y == 0] = -1

    trees, a = adaBoostTrain(X, y,tree_num=3)
    pltAdaBoostDecisionBound(X, y, trees, a)
    trees, a  = adaBoostTrain(X, y, tree_num=5)
    pltAdaBoostDecisionBound(X, y, trees, a)
    trees, a  = adaBoostTrain(X, y, tree_num=11)
    pltAdaBoostDecisionBound(X, y, trees, a)
