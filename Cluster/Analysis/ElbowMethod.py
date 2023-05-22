from pathlib import Path
from Cluster.KMeans import K_means
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import pickle
import gzip
import torch


#读取数据
DATA_PATH = Path("../../data")
PATH = DATA_PATH / "MNIST"
FILENAME = 'mnist.pkl.gz'
with gzip.open((PATH/FILENAME).as_posix(),"rb") as f:
    ((x_train,y_train),(x_valid,y_valid),_) = pickle.load(f,encoding="latin-1")

    x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_valid = x_valid.numpy()
    y_valid = y_valid.numpy()
#读取数据完成
print("load over")

#仅选用5000张图片，作肘部法则运算
m = 5000
n_clusters = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,]
dist_cost = []
train = x_train[:m,:]
labels= y_train[:m]
dist = 0
for i in n_clusters:
    #聚类
    kmeans_algorithm = K_means(i, 100, train, labels)
    kmeans_algorithm.main_cluster()

    for j in range(train.shape[0]):
        #classification[i]保存的是第i个样本的中心点 （m,1）
        #distance保存的是每一个样本到每一个中心点的距离 (m,n_cluster)

        #cost function
        dist = dist + kmeans_algorithm.distances[j,kmeans_algorithm.classification[j]]
    print("dist:{},dist_exp:{}".format(dist,np.exp(dist/m)))
    dist_cost.append(np.exp(dist/m))
    dist = 0

print(dist_cost)
#数据作归一化处理  线性归一化
dist_cost = (dist_cost - np.min(dist_cost))/(np.max(dist_cost)-np.min(dist_cost))
#activation function
#dist_cost = np.tanh(dist_cost)

#输出图像
print(dist_cost)
plt.plot(n_clusters[:-1],dist_cost[:-1], marker=".")
plt.xlabel("# of clusters")
plt.ylabel("cost function")
plt.savefig('../../Figure/ElbowMethod.jpg')
plt.show()
