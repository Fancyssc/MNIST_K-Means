from pathlib import Path
from sklearn.metrics import confusion_matrix
from Cluster.KMeans import K_means
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import gzip
import torch
import time



# 数据集路径
DATA_PATH = Path("../data")
PATH = DATA_PATH / "MNIST"


#URL MNIST线上下载地址
# URL = "https://deeplearning.net/data/mnist/"
FILENAME = 'mnist.pkl.gz'

#如果MNIST本地数据集不存在，下载数据集
# if not(PATH/FILENAME).exists():
#     content = requests.get(URL+FILENAME).content
#     (PATH/FILENAME).open("wb").write(content)


#解压数据集
## gzip用于打开文件,rb分别标识“只读”和“二进制”     as_posix()转python路径为linux路径
## pickel.dump: python对象转二进制对象， pickel.load:反之
with gzip.open((PATH/FILENAME).as_posix(),"rb") as f:
    ((x_train,y_train),(x_valid,y_valid),_) = pickle.load(f,encoding="latin-1")

# #显示图片，验证读成功读取
# plt.imshow(x_train[0].reshape(28,28),cmap="gray")
# plt.show()
# print(x_train.shape)

#改成torch tensor形式
x_train,y_train,x_valid,y_valid= map(torch.tensor,(x_train,y_train,x_valid,y_valid))

#由tensor转型成numpy
x_train= x_train.numpy()
y_train= y_train.numpy()
x_valid= x_valid.numpy()
y_valid= y_valid.numpy()


#参数定义
m = 50000
#由Elbow method确定
n_clusters = 20
#选取其中的150张图片，放入train中：
train = x_train[:m,:]
labels= y_train[:m]


time_start = time.time()
# 调用K-means
kmeans_algorithm=K_means(n_clusters,50,train,labels)
kmeans_algorithm.main_cluster()
time_end= time.time()
print("K_means costs: {}".format(time_end-time_start)+" seconds.")
# 聚类精度检查
# 构建类和标签的映射
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


# 用于绘制热度图
heatmap = np.zeros([10,10])
for i in range(train_preds.shape[0]):
    heatmap[train_preds[i]][labels[i]] = heatmap[train_preds[i]][labels[i]] + 1
sns.heatmap(heatmap, annot=True, fmt='.0f',cmap="Reds")
plt.xlabel("predicted value")
plt.ylabel("label value")
plt.title("K-Means")
plt.savefig("../Figure/Heatmap_pure.jpg")
plt.show()

print("Cluster Accuracy：{}".format(np.sum(train_preds == labels) / labels.shape[0]))

