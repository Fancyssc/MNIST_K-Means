from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from Cluster.KMeans import K_means
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import pickle
import gzip
import torch

#载入数据
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

#实例化PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x_train)

plt.figure(1)
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], s=5, c=y_train, cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
plt.savefig("../../Figure/PCA.jpg")
plt.title('PCA', fontsize=18)

plt.figure(2)
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(x_train)
tsne = TSNE(random_state = 42, n_components=2, verbose=0, perplexity=40, n_iter=300).fit_transform(pca_result_50)
plt.scatter(tsne[:, 0], tsne[:, 1], s=5, c=y_train, cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.savefig("../../Figure/t-SNE.jpg")
plt.title('t-SNE', fontsize=18)

plt.show()



