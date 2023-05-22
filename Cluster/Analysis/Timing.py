from pathlib import Path
from Cluster.KMeans import K_means
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
n_clusters = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
time_cost = []
train = x_train[:m,:]
labels= y_train[:m]
dist = 0
for i in n_clusters:
    #聚类
    time_start = time.time()
    kmeans_algorithm = K_means(i, 100, train, labels)
    kmeans_algorithm.main_cluster()
    time_end = time.time()
    time_cost.append(time_end - time_start)


#输出图像
print(time_cost)
plt.plot(n_clusters,time_cost, marker=".")
plt.show()
