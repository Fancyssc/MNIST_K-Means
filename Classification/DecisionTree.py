from pathlib import Path
from sklearn import tree
import time
import pickle
import gzip
import torch
import matplotlib.pyplot as plt
import numpy as np

#Test

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


#定义决策树
classifier = tree.DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=21, min_samples_split=3,random_state=40)
x_train = x_train.reshape(50000,784)
x_valid = x_valid.reshape(10000,784)

classifier.fit(x_train,y_train)
score = classifier.score(x_valid,y_valid)
print(score)

