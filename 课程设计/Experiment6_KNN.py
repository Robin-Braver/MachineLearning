"""编程实现KNN分类器，选择两个UCI数据集进行实验及测试"""
import numpy as np
import operator
import utils

def KNN_Predict_One(train_data,data,K=5):
    m,n=train_data.shape
    distance_label=np.zeros((m,2))
    distance_label[:,1]=train_data[:,-1]

    # 计算距离
    for i in range(m):
        temp_sum=0
        for j in range(n-1):
            temp_sum+=(train_data[i,j]-data[j])**2
        distance_label[i,0]=temp_sum**0.5

    #排序
    distance_index=distance_label[:,0].argsort()

    #统计K近邻
    label_count = {}
    for i in range(K):
        temp_label=int(distance_label[distance_index[i],1])
        label_count[temp_label]=label_count.get(temp_label,0)+1
    max_label_count=sorted(label_count.items(),key=operator.itemgetter(1),reverse=True)

    predict_label=max_label_count[0][0]

    return predict_label


def KNN_Predict(train_data,test_data,K=5):
    pred_label=[]
    for data in test_data:
        pred_label.append(KNN_Predict_One(train_data,data,K))
    return pred_label



if __name__ == '__main__':
    # 读取数据并划分训练集以及测试集

    filename="data/heart.dat"#haberman.data【3,0.3】\heart.dat【】
    dataSet = utils.getDataSet(filename)
    train_data, test_data = utils.splitDataSet1(dataSet, test_size=0.3)

    test_data_data=test_data[:,:-1]
    real_label=list(test_data[:,-1])
    for i in range(len(real_label)):
        real_label[i]=int(real_label[i])

    # KNN算法测试[1,2,3,4,5,6,7,8,9,10,15,17,23][3,7,11][3,6,11]
    for K in [3,5,7,11]:
        pred=KNN_Predict(train_data=train_data,test_data=test_data_data,K=K)
        print("------------"+filename+"---KNN算法---K="+str(K)+"------------")
        # print("true:", real_label)
        # print("pred:", pred_label)
        acc, p, r, f1 = utils.calAccuracy(pred, real_label)
        print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))

    # 读取数据并划分训练集以及测试集
    filename="data/haberman.data"#haberman.data【3,0.3】\heart.dat【】
    dataSet = utils.getDataSet(filename)
    train_data, test_data = utils.splitDataSet1(dataSet, test_size=0.3)
    print()
    test_data_data=test_data[:,:-1]
    real_label=list(test_data[:,-1])
    for i in range(len(real_label)):
        real_label[i]=int(real_label[i])
    for K in [3,5,7,11]:
        pred=KNN_Predict(train_data=train_data,test_data=test_data_data,K=K)
        print("------------"+filename+"---KNN算法---K="+str(K)+"------------")
        # print("true:", real_label)
        # print("pred:", pred_label)
        acc, p, r, f1 = utils.calAccuracy(pred, real_label)
        print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))



