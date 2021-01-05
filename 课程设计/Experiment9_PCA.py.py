"""实验九：自学Sklearn的PCA函数使用方法，对Yale人脸数据集进行降维，观察并分析前20个特征向量对应的图像"""
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os



def Data_to_DataSet_20(directorypath='yalefaces',save=False,m=243,n=320,c=165):
    file_list = os.listdir(directorypath)
    img_dataset = np.zeros((c,m,n))
    index=0
    for file in file_list:
        file1 = directorypath + '/' + file
        img = mpimg.imread(file1)
        img_dataset[index,:,:]=img
        index+=1
    if save==True:
        file2=directorypath+'_dataset_20.npy'
        np.save(file2,img_dataset)
    return img_dataset

def get_dataset_20(filename='yalefaces_dataset_20.npy'):
    dataset=np.load(filename)
    return dataset

def plot_raw_grey_image_20(data_set,index):
    img=data_set[index-1,:,:]

    plt.imshow(img, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.show()

def apply_pca(dataset_pca,n_components=20):
    pca = PCA(n_components=n_components)
    dataset_pca1 = pca.fit_transform(dataset_pca)
    dataset_pca_recover=pca.inverse_transform(dataset_pca1)
    feature_vector = pca.components_

    return dataset_pca1,dataset_pca_recover,feature_vector


def save_plot_grey_image_20(data_set,index,directorypath='PCA20',n_components=20):
    img=data_set[index-1,:,:]

    plt.imshow(img, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    file2 = directorypath+ '/PCA' + str(n_components) + '_' + str(index) + '.jpg'
    mpimg.imsave(file2, img, cmap='gray')

if __name__=='__main__':

    directorypath1 = 'data/yalefaces1'
    m, n, c = 100, 100, 165  # yalefaces:(243,320,165);yalefaces1:(100,100,165)
    n_components = 20
    # 165张图片转换成npy数据集(100,100,165)
    data_set = Data_to_DataSet_20(directorypath=directorypath1, save=False, m=m, n=n, c=c)

    # 改变数据集形状(100,100,165)为(165,100,100)，用于PCA降维
    dataset_pca = data_set.reshape((c, m*n))
    # 对数据集进行降维
    dataset_pca20,dataset_pca_recover,feature_vector = apply_pca(dataset_pca=dataset_pca, n_components=n_components)


    feature_vector = feature_vector.reshape((n_components, m, n))

    plot_one_image = np.zeros((500,400))
    index = 0
    for w in range(0,5):
        for h in range(0,4):
            plot_one_image[w*100:(w+1)*100,h*100:(h+1)*100] = feature_vector[index,:,:]
            index += 1
    plt.imshow(plot_one_image, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    # # 绘制并保存数据集中的第index张灰度人脸图片
    # for i in range(20):
    #     save_plot_grey_image_20(data_set=feature_vector, index=i+1,directorypath='result/PCA_20',n_components=n_components)

    print("Execute successfully!")