'''编程实现 基于信息熵进行划分选择的决策树算法 （ 包括ID3,
C4.5 两种算法 ），并为表4.3中的数据生成一颗决策树。'''
import math
import random
from collections import Counter
import pandas as pd
import numpy as np
import pygraphviz as pyg

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
        if pred[i] == 1 and label[i]==1:
            TP += 1
        elif pred[i] == 1 and label[i]==0:
            FP += 1
        elif pred[i] == 0 and label[i]==1:
            FN += 1
        elif pred[i] == 0 and label[i]==0:
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



def get_dataset(filename ="table4.3.txt"):
    """
   get_dataset：从文件中读取数据，将其转化为指定格式的dataset和col_name
   :returns  dataset:数据+标签 col_name:列名，即有哪些属性
   """
    df = pd.read_csv(filename)
    #下面分别提取标签和数据:一行对应一个数据
    dataset = df.values.tolist()
    col_name = df.columns.values[0:-1].tolist()#剔除最后一列
    return dataset,col_name

class Node(object):
    """
    决策树节点类
    """
    def __init__(self):
        self.attr = None     # 决策节点属性
        self.value = None    # 节点与父节点的连接属性
        self.isLeaf = False  # 是否为叶节点 T or F
        self.label = 0    # 叶节点对应标签
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
        self.label= label


class DataSet(object):
    """
    数据集类
    """
    def __init__(self, data_set, col_name):
        """
        :param col_name 列名 list[]
        :param data_set 数据集 list[list[]]:
        """
        self.num = len(data_set) #获得数据集大小
        self.data_set = data_set
        self.data = [example[:-1] for example in data_set if self.data_set[0]]  #提取数据剔除标签
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

    def cal_ent(self):
        """
        计算信息熵
        :return:float 信息熵
        """
        class_dict = Counter(self.labels)
        # 计算信息熵
        ent = 0
        for key in class_dict.keys():
            p = float(class_dict[key]) / self.num
            ent -= p*math.log(p, 2)
        return ent

    def split_continuous_feature(self, index, mid):
        """
        划分连续型数据
        :param index:是哪一列特征
        :param mid:考察的中位数
        :return:经过DataSet类封装的数据集
        """
        left_feature_list = []
        right_feature_list = []
        for example in self.data_set:#遍历每一行数据
            if example[index] <= mid:
                left_feature_list.append(example)  # 此处不用del第axis个标签，连续变量的属性可以作为后代节点的划分属性
            else:
                right_feature_list.append(example)
        return DataSet(left_feature_list, self.col_name), DataSet(right_feature_list, self.col_name)

    def choose_byGain(self):
        """
        根据信息增益选择最佳列属性划分
        :return: 最佳划分属性索引，int
        """
        num_features = len(self.data[0])#特征数量
        max_Gain = -999#最大增益率
        best_feature_index = -1
        best_col = ""
        for index in range(num_features):
            new_ent = 0
            feature_list = [example[index] for example in self.data]#提取每一行特征
            if type(feature_list[0]).__name__ == "float":   # 如果是连续值
                sorted_feature_list = sorted(feature_list)#对连续值排序
                split_list = []
                for k in range(self.num - 1):#考察选取哪一个中位数点
                    mid_feature = (sorted_feature_list[k] + sorted_feature_list[k+1])/2#计算中位数
                    split_list.append(mid_feature)
                    left_feature_list, right_feature_list = self.split_continuous_feature(index, mid_feature)
                    p_left = left_feature_list.num / float(self.num)
                    p_right = right_feature_list.num / float(self.num)
                    new_ent = p_left*left_feature_list.cal_ent() + p_right*right_feature_list.cal_ent()#划分后只有两个类别
                    gain = self.cal_ent() - new_ent  # 计算增益率
                    if gain > max_Gain:
                        max_Gain = gain
                        best_feature_index = index
                        best_col = self.col_name[best_feature_index] + "<=" + str(mid_feature)
            else:
                unique_feature = np.unique(feature_list)#剔除重复出现的特征减少计算量
                for feature in unique_feature:
                    sub_data_set = self.group_feature(index,feature)
                    p = sub_data_set.num/float(self.num)#同一个个数比例
                    new_ent += p*sub_data_set.cal_ent()
                gain = self.cal_ent()-new_ent#计算增益率
                if gain > max_Gain:
                    max_Gain = gain
                    best_feature_index = index
                    best_col = self.col_name[best_feature_index]
        return best_feature_index, best_col

    def choose_byGain_ratio(self):
        """
        根据信息增益率选择最佳列属性划分
        :return: 最佳划分属性索引，int
        """
        gain_list = []#记录各个属性的信息增益
        gainRatio_list = []#记录各个属性的增益率
        col_list = []
        num_features = len(self.data[0])#特征数量
        best_col = ""
        for index in range(num_features):
            new_ent = 0
            feature_list = [example[index] for example in self.data]#提取每一行特征
            if type(feature_list[0]).__name__ == "float":   # 如果是连续值
                sorted_feature_list = sorted(feature_list)#对连续值排序
                split_list = []
                maxGain = -999
                for k in range(self.num - 1):#考察选取哪一个中位数点
                    mid_feature = (sorted_feature_list[k] + sorted_feature_list[k+1])/2#计算中位数
                    split_list.append(mid_feature)
                    left_feature_list, right_feature_list = self.split_continuous_feature(index, mid_feature)
                    p_left = left_feature_list.num / float(self.num)
                    p_right = right_feature_list.num / float(self.num)
                    new_ent = p_left*left_feature_list.cal_ent() + p_right*right_feature_list.cal_ent()#划分后只有两个类别
                    gain = self.cal_ent() - new_ent  # 计算
                    if gain>maxGain:
                        maxGain = gain
                        best_feature_index = index
                        best_col = self.col_name[best_feature_index] + "<=" + str(mid_feature)
                        IV =p_left*math.log(p_left,2)+math.log(p_right,2)*p_right
                        IV = 0-IV#取相反数
                        Gain_ratio = gain/IV#计算信息增益
                gain_list.append(maxGain)
                gainRatio_list.append(Gain_ratio)
                col_list.append(best_col)
            else:
                unique_feature = np.unique(feature_list)#剔除重复出现的特征减少计算量
                IV = 0
                for feature in unique_feature:
                    sub_data_set = self.group_feature(index,feature)
                    p = sub_data_set.num/float(self.num)#同一个个数比例
                    new_ent += p*sub_data_set.cal_ent()
                    IV += p*math.log(p,2)
                gain = self.cal_ent()-new_ent#计算信息增益
                if IV==0:
                    IV=0.001
                IV = 0 - IV  # 取相反数
                Gain_ratio = gain / IV#计算增益率
                gain_list.append(gain)
                gainRatio_list.append(Gain_ratio)
                col_list.append(self.col_name[index])
        av_gain = np.average(gain_list)#求均值
        dic_table = {}#信息增益和增益率的对照字典，key为增益率，值为信息增益和index的元祖
        for i in range(num_features):
            dic_table[gainRatio_list[i]] = (gain_list[i],i)
        gainRatio_list.sort(reverse=True)#对增益率排序
        for ratio in gainRatio_list:
            if dic_table[ratio][0]>=av_gain:#高于平均值
                index = dic_table[ratio][1]
                return index, col_list[index]
            else:
                continue


class DecisionTree(object):
    """
    决策树
    """
    def __init__(self, data_set:DataSet,strategy):
        self.root = self.bulidTree(data_set,strategy)#头结点
    def bulidTree(self,data_set:DataSet,strategy):
        """
        :param data_set: 建立树用到的数据集
        :param strategy:选择建立树的策略包含--信息增益:gain --增益率gain-ratio
        :return:建立树的头结点
        """
        node = Node()
        if data_set.is_OneClass() != -1:  # 如果数据集中样本属于同一类别
            node.isLeaf = True   # 标记为叶子节点
            node.label = data_set.is_OneClass()  # 标记叶子节点的类别为此类
            return node
        if len(data_set.data) == 0 or data_set.is_sameFeature():  # 如果数据集为空或者数据集的特征向量相同
            node.isLeaf = True   # 标记为叶子节点
            node.label = data_set.get_mostLabel()   # 标记叶子节点的类别为数据集中样本数最多的类
            return node
        if strategy=='gain_ratio':
            best_feature_index, best_col = data_set.choose_byGain_ratio()  # 最佳划分数据集的标签索引,最佳划分数据集的列名
        elif strategy=='gain':
            best_feature_index, best_col = data_set.choose_byGain()
        else:
            print("输入策略有误！！！")
            return node
        node.attr = best_col   # 设置非叶节点的属性为最佳划分数据集的列名
        if u'<=' in node.attr:  #如果当前属性是连续属性
            mid_value = float(best_col.split('<=')[1])  # 获得比较值
            left_data_set, right_data_set = data_set.split_continuous_feature(best_feature_index, mid_value)
            left_node = Node()
            right_node = Node()
            '''分别求左右子树'''
            if left_data_set.num == 0:  # 如果子数据集为空
                left_node.isLeaf = True  # 标记新节点为叶子节点
                left_node.label = data_set.get_mostLabel()  # 类别为父数据集中样本数最多的类
            else:
                left_node = self.bulidTree(left_data_set,strategy)#递归生成新的树
            if right_data_set.num == 0:  # 如果子数据集为空
                right_node.isLeaf = True  # 标记新节点为叶子节点
                right_node.label = data_set.get_mostLabel()  # 类别为父数据集中样本数最多的类
            else:
                right_node = self.bulidTree(right_data_set,strategy)
            left_node.value = '小于等于'
            right_node.value = '大于'
            node.children.append(left_node)#记录孩子
            node.children.append(right_node)#记录又孩子
        else:
            feature_set = np.unique([example[best_feature_index] for example in data_set.data]) #最佳划分标签的可取值集合
            for feature in feature_set:
                new_node = Node()
                sub_data_set = data_set.group_feature(best_feature_index, feature)   # 划分数据集并返回子数据集
                if sub_data_set.num == 0:# 如果子数据集为空
                    new_node.isLeaf = True # 标记新节点为叶子节点
                    new_node.label = data_set.get_mostLabel()   # 类别为父数据集中样本数最多的类
                else:
                    new_node = self.bulidTree(sub_data_set,strategy)
                new_node.value = feature  # 设置节点与父节点间的连接属性
                new_node.parent = node  # 设置父节点
                node.children.append(new_node)
        return node

    def save(self, filename):
        g = pyg.AGraph(strict=False, directed=True)
        g.add_node(self.root.attr, fontname="Microsoft YaHei")
        self._save(g, self.root)
        g.layout(prog='dot')
        g.draw(filename)

    def _save(self, graph, root):
        if len(root.children) > 0:
            for node in root.children:
                if node.attr != None:
                    graph.add_node(node.attr, fontname="Microsoft YaHei")
                    graph.add_edge(root.attr, node.attr, label=node.value, fontname="Microsoft YaHei")
                else:
                    graph.add_node(node.label, shape="box", fontname="Microsoft YaHei")
                    graph.add_edge(root.attr, node.label, label=node.value, fontname="Microsoft YaHei")
                self._save(graph, node)

    def predict_one(self, test_x, node, col_name) -> int:
        if (len(node.children) == 0):
            return node.label
        else:
            if u'<=' not in node.attr:  # 处理离散数据
                index = col_name.index(node.attr)
                for chil in node.children:
                    if chil.value == None:  # 是一个叶子节点
                        return chil.label
                    if chil.value == test_x[index]:
                        return self.predict_one(test_x, chil, col_name)
            else:
                col_ = node.attr.split("<=")[0]
                value = eval(node.attr.split("<=")[1])
                index = col_name.index(col_)  # 这个元素的索引是啥
                if test_x[index] <= value:  # 小于划分节点，在左子树
                    return self.predict_one(test_x, node.children[0], col_name)
                else:
                    return self.predict_one(test_x, node.children[1], col_name)

    def predict(self, test_data, col_name):
        pred = []
        for data in test_data:
            pred_y = self.predict_one(data, self.root, col_name)
            pred.append(pred_y)
        return pred

dataset,col_name = get_dataset()
random.shuffle(dataset)
train_data, test_data = dataset[0:12],dataset[12:17]
test_label = [test_data[i][-1] for i in range(len(test_data))]
Train_Data_set = DataSet(train_data,col_name)
print("*"*10+"测试集样本情况"+"*"*10)
for data in test_data:
    print(data)

print("*"*10+"信息增益率"+"*"*10)
decisionTree = DecisionTree(Train_Data_set,strategy ='gain_ratio')
decisionTree.save('watermelon_gainRatio.jpg')
pred = decisionTree.predict(test_data,col_name)
print("预测值：",pred)
acc, p, r, f1 = calAccuracy(pred,test_label)
print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc,p,r,f1))

print("*"*10+"信息增益"+"*"*10)
decisionTree = DecisionTree(Train_Data_set,strategy ='gain')
decisionTree.save('watermelon_gain.jpg')
pred = decisionTree.predict(test_data,col_name)
print("预测值：",pred)
acc, p, r, f1 = calAccuracy(pred,test_label)
print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc,p,r,f1))

