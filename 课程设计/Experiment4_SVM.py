"""在西瓜数据集3.0a上分别用线性核和高斯核训练一个SVM."""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from 课程设计.utils import *

def SVM_revalue(ori_y):
    """
   SVM_revalue函数：将y值转换为仅在-1和1上取值
   :param ori_y:原始的y值
   :returns  modify_y：经过处理后的y值
   """
    max_y = max(ori_y)
    modify_y = []
    for y in ori_y:
        if y==max_y:
            modify_y.append(1)
        else:
            modify_y.append(-1)
    return np.array(modify_y)

class SVC:
    def __init__(self,max_iter=100,C=1,kernel='rbf',gamma=1):
        self.b=0.
        self.alpha=None
        self.max_iter=max_iter
        self.C=C#C是惩罚系数，即对误差的宽容度。c越高，说明越不能容忍出现误差, 容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差
        self.kernel=kernel
        self.K=None
        self.X=None
        self.y=None
        if kernel=='rbf':
            self.sigma=1/(gamma*2)**0.5
        pass

    def kernel_func(self,kernel,x1,x2):
        """
      kernel_func函数：核函数
       :param: kernel:核函数名
       :param x1:数据1
       :param x2:数据2
       :returns 核函数处理后的结果
      """
        if kernel=='linear':
            return x1.T.dot(x2)
        elif kernel=='rbf':
            return np.exp(-(np.sum((x1-x2)**2))/(2*self.sigma*self.sigma))

    def computeK(self,X,kernel):
        """
       computeK函数：计算核矩阵
        :param: X:原始样本
        :param kernel:选择的核函数
        :returns  K：计算出的核函数
       """
        m=X.shape[0]
        K=np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                if i<=j:
                    K[i,j]=self.kernel_func(kernel,X[i],X[j])
                else:
                    K[i,j]=K[j,i]
        return K

    def compute_u(self,X,y):
        """
        compute_u函数：计算预测值
         :param: X:原始样本
         :param y:真实值
         :returns  u：预测值
        """
        u = np.zeros((X.shape[0],))
        for j in range(X.shape[0]):
            u[j]=np.sum(y*self.alpha*self.K[:,j])+self.b
        return u

    def checkKKT(self,u,y,i):
        """
        checkKKT函数：检查是否符合KKT条件
         :param: u:预测值
         :param y:真实值
         :param i：标签的下标，即第几个标签
         :returns  True：符合 or False：不符合
        """
        if self.alpha[i]<self.C and y[i]*u[i]<=1:
            return False
        if self.alpha[i]>0 and y[i]*u[i]>=1:
            return False
        if (self.alpha[i]==0 or self.alpha[i]==self.C) and y[i]*u[i]==1:
            return False
        return True

    def fit(self,X,y):
        """
        fit基于SMO算法优化，如果所有变量的解都满足此最优化问题的KKT条件，那么这么最优化问题的解就得到了。
        :param X: 数据集
        :param y: 标签
        """
        self.X=X
        self.y=y
        self.K=self.computeK(X,self.kernel)
        self.alpha = np.zeros((X.shape[0],))
        for _ in range(self.max_iter):
            u = self.compute_u(X, y)
            finish=True
            for i in range(X.shape[0]):
                if not self.checkKKT(u,y,i):
                    #检查训练样本中每个点 [公式] 是否满足KKT条件，如果不满足，则它对应的 [公式] 可以被优化，
                    # 然后随机选择另一个变量 [公式] 进行优化。
                    finish=False
                    #保证选取到的不是目前已选中的一个
                    y_indices=np.delete(np.arange(X.shape[0]),i)
                    j=y_indices[int(np.random.random()*len(y_indices))]
                    eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
                    # 计算学习率
                    if eta == 0:
                        continue
                    #计算误差项
                    E_i=np.sum(self.alpha*y*self.K[:,i])+self.b-y[i]
                    E_j=np.sum(self.alpha*y*self.K[:,j])+self.b-y[j]
                    #对alpha进行修剪，考虑对alpha j优化，计算上下边界L，H
                    if y[i]!=y[j]:
                        L=max(0,self.alpha[j]-self.alpha[i])
                        H=min(self.C,self.C+self.alpha[j]-self.alpha[i])
                    else:
                        L=max(0,self.alpha[j]+self.alpha[i]-self.C)
                        H=min(self.C,self.alpha[j]+self.alpha[i])
                    #计算新的alpha2
                    alpha2_new_unc=self.alpha[j]+y[j]*(E_i-E_j)/eta
                    alpha2_old=self.alpha[j]
                    alpha1_old=self.alpha[i]
                    #考虑约束条件得到最后解
                    if alpha2_new_unc>H:
                        self.alpha[j]=H
                    elif alpha2_new_unc<L:
                        self.alpha[j]=L
                    else:
                        self.alpha[j]=alpha2_new_unc
                    #进一步计算alpha i
                    self.alpha[i]=alpha1_old+y[i]*y[j]*(alpha2_old-self.alpha[j])
                    # 更新阈值b
                    b1_new=-E_i-y[i]*self.K[i,i]*(self.alpha[i]-alpha1_old)-y[j]*self.K[j,i]*(self.alpha[j]-alpha2_old)+self.b
                    b2_new=-E_j-y[i]*self.K[i,j]*(self.alpha[i]-alpha1_old)-y[j]*self.K[j,j]*(self.alpha[j]-alpha2_old)+self.b
                    if self.alpha[i]>0 and self.alpha[i]<self.C:
                        self.b=b1_new
                    elif self.alpha[j]>0 and self.alpha[j]<self.C:
                        self.b=b2_new
                    else:
                        self.b=(b1_new+b2_new)/2
                if finish:
                    break

    def predict(self,X):
        """
        predict:预测值
        :param X: 数据集
        :return y_preds:预测值
        """
        y_preds=[]
        for i in range(X.shape[0]):
            K=np.zeros((len(self.y),))
            support_indices=np.where(self.alpha>0)[0]
            for j in support_indices:
                K[j]=self.kernel_func(self.kernel,self.X[j],X[i])
            y_pred=np.sum(self.y[support_indices]*self.alpha[support_indices]*K[support_indices].T)
            y_pred+=self.b
            if y_pred>=0:
                y_preds.append(1)
            else:
                y_preds.append(-1)
        return np.array(y_preds)

def test(filename,test_size=0.3,seed=22):
    dataSet_name = filename.split('/')[1].split('.')[0]
    X, Y_, _ = getData(filename)
    Y = SVM_revalue(Y_)
    print("\033[31m------------------------" + dataSet_name + "------------------------\033[0m")
    train_data, train_label, test_data, test_label = splitDataSet(X, Y, test_size=test_size,seed=seed)
    # print("-------------自己代码：线性核-------------------------")
    svc = SVC(max_iter=100, kernel='linear', C=20)
    svc.fit(train_data, train_label)
    pred_y = svc.predict(test_data)
    # print(pred_y)
    # print(test_label)
    acc, p, r, f1 = calAccuracy(pred_y, test_label)
    # accuracy.append(acc)
    # ps.append(p)
    # rs.append(r)
    # F1s.append(f1)
    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))
    print("-------------自己代码：rbf-------------------------")
    svc = SVC(max_iter=100, kernel='rbf', C=2, gamma=20)
    svc.fit(train_data, train_label)  # 磨合模型
    pred_y = svc.predict(test_data)
    acc, p, r, f1 = calAccuracy(pred_y, test_label)


    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))
    print("-------------sklearn：线性核--------------------------")
    from sklearn import svm, metrics
    clf = svm.SVC(kernel='linear')  # .SVC（）就是 SVM 的方程，参数 kernel 为线性核函数
    clf.fit(train_data, train_label)
    pred_y = clf.predict(test_data)
    # print(pred_y)
    # print(test_label)
    acc, p, r, f1 = calAccuracy(pred_y, test_label)
    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))
    print("-------------sklearn：rbf--------------------------")
    clf = svm.SVC(kernel='rbf', gamma=20,C=2)  # .SVC（）就是 SVM 的方程，参数 kernel 为线性核函数
    ''' gamma是选择RBF函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，
    gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。'''
    clf.fit(train_data, train_label)
    pred_y = clf.predict(test_data)
    acc, p, r, f1 = calAccuracy(pred_y, test_label)
    print("正确率:{:.2%}\t查准率:{:.4f}\t查全率:{:.4f}\tF1:{:.4f}".format(acc, p, r, f1))


if __name__=='__main__':
    test_size = 0.3
    test("data/haberman.data",test_size,2)
    test("data/heart.dat", test_size,2)
