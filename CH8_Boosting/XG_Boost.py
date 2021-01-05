from sklearn import datasets
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score, recall_score

# 加载数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print("Train data length:", len(X_train))
print("Test data length:", len(X_test))

# 转换为DMatrix数据格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数
parameters = {
    'eta': 0.3,
    'silent': True,  # option for logging
    'objective': 'multi:softprob',  # error evaluation for multiclass tasks
    'num_class': 3,  # number of classes to predic
    'max_depth': 3  # depth of the trees in the boosting process
}
num_round = 20  # the number of training iterations
# 模型训练
bst = xgb.train(parameters, dtrain, num_round)
# 模型预测
preds = bst.predict(dtest)
print(preds[:5])
# 选择表示最高概率的列
best_preds = np.asarray([np.argmax(line) for line in preds])
print(best_preds)
# 模型评估
print(precision_score(y_test, best_preds, average='macro'))  # 精准率
print(recall_score(y_test, best_preds, average='macro'))  # 召回率