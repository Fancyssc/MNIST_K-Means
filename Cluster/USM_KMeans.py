from pathlib import Path
from skimage.filters import gaussian
import time
from Cluster.KMeans import K_means
import matplotlib.pyplot as plt
import pickle
import gzip
import torch
import numpy as np

#读取数据
DATA_PATH = Path("../data")
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
time_start = time.time()

#锐化
for i in range(x_train.shape[0]):
    test_image = x_train[i].reshape(28,28)
    #高斯核抓取低频
    img = test_image * 1.0
    gauss_out = gaussian(img, sigma=3)
    # alpha 0 - 5
    alpha = 5
    img_out = (img - gauss_out) * alpha + img
    img_out = img_out/255.0
    # 饱和处理
    mask_1 = img_out  < 0
    mask_2 = img_out  > 1
    img_out = img_out * (1-mask_1)
    img_out = img_out * (1-mask_2) + mask_2

    # for e_i in range(28):
    #     for e_j in range(28):
    #         if (img_out[e_i][e_j] < 0.01):
    #             img_out[e_i][e_j] = 0
    x_train[i] = img_out.reshape(1,784)
time_end = time.time()

print("USM Sharpening costs: {}".format(time_end-time_start)+".seconds.")
#参数定义
m = 50000
n_clusters = 20

train = x_train[:m,:]
labels= y_train[:m]

time_start = time.time()
# 调用K-means
kmeans_algorithm=K_means(n_clusters,50,train,labels)
kmeans_algorithm.main_cluster()
time_end= time.time()
print("K_means costs: {}".format(time_end-time_start)+" seconds.")

label_num = np.zeros((n_clusters, 10),dtype=int)
for i in range(kmeans_algorithm.classification.shape[0]):
    prediction = int(kmeans_algorithm.classification[i])
    truth = int(labels[i])
    label_num[prediction][truth] += 1

# 查看n_cluster 和 label 的对应关系
# 对于每一个cluster, 选择对应数量最多的label类
label_to_num = label_num.argmax(axis=1)     #label_to_num (1,30)
set(label_to_num)
print("label_to_num")
print(label_to_num)
train_preds = np.zeros(labels.shape,dtype=int)
for i in range(train_preds.shape[0]):
    train_preds[i] = label_to_num[kmeans_algorithm.classification[i]]

print("Cluster Accuracy：{}".format(np.sum(train_preds == labels) / labels.shape[0]))