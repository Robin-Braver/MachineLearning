import xgboost as xgb
import numpy as np
import utils

def XGboost_revalue(ori_y):
    """
   XGboost_revalue函数：将y值转换为0 1
   :param ori_y:原始的y值
   :returns  modify_y：经过处理后的y值
   """
    max_y = max(ori_y)
    modify_y = []
    for y in ori_y:
        if y==max_y:
            modify_y.append(1)
        else:
            modify_y.append(0)
    return np.array(modify_y)

# accl = []
# f1l = []
def test(filename):
    X, Y_,_ = utils.getData(filename)
    Y = XGboost_revalue(Y_)
    dataSet_name = filename.split('/')[1].split('.')[0]
    print("------------------------" + dataSet_name + "------------------------")
    train_data, train_label, test_data, test_label = utils.splitDataSet(X, Y, test_size=0.3)
    # 转换为DMatrix数据格式
    dtrain = xgb.DMatrix(train_data, label=train_label)
    dtest = xgb.DMatrix(test_data, label=test_label)
    # 设置参数
    parameters = {
        'eta': 0.01,
        'subsample': 0.75,
        'objective': 'multi:softmax',  # error evaluation for multiclass tasks
        'num_class': 2,  # number of classes to predic
        'max_depth':8  # depth of the trees in the boosting process
    }
    num_round = 500  # the number of training iterations
    bst = xgb.train(parameters, dtrain, num_round)
    preds = bst.predict(dtest)#输出的是概率
    acc, p, r, f1 = utils.calAccuracy(preds,test_label)
    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc,p,r,f1))
    # accl.append(acc)
    # f1l.append(f1)

if __name__ == "__main__":
    for i in range(1):
        test("data/haberman.data")
        test("data/heart.dat")
    # for i in accl:
    #     print(i)
    # for i in f1l:
    #     print(i)
