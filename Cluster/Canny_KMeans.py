from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
from Cluster.KMeans import K_means
from skimage.feature import canny
import time
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
    img = x_train[i].reshape(28,28)
    img_save = x_train[i].reshape(28,28)
    img_out = canny(img,sigma=0.5)
    x_train[i] = img_out.reshape(1,784) + img_save.reshape(1,784)
time_end = time.time()

print("Canny costs: {}".format(time_end-time_start)+".seconds.")
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

heatmap = np.zeros([10,10])
for i in range(train_preds.shape[0]):
    heatmap[train_preds[i]][labels[i]] = heatmap[train_preds[i]][labels[i]] + 1
sns.heatmap(heatmap, annot=True, fmt='.0f',cmap="Reds")
plt.xlabel("predicted value")
plt.ylabel("label value")
plt.title("K-Means with Canny")
plt.savefig("../Figure/Heatmap_canny.jpg")
plt.show()

print("Cluster Accuracy：{}".format(np.sum(train_preds == labels) / labels.shape[0]))