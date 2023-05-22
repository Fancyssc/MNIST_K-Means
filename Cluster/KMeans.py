import numpy as np
class K_means:
    def __init__(self, k, maxiter, data, labels):
        self.k = k  # 分类类数
        self.maxiter = maxiter  # 最大迭代次数
        # data.shape[0] -> MNIST 50000张  m
        # data.shape[1] -> 28*28= 784维
        self.data = data     # 输入数据(50000*28*28)
        self.labels = labels  # 标签
        # 记录距离(m,k)
        self.distances = np.zeros((self.data.shape[0], self.k))
        # 保存中心点每个类中心点
        self.centre = np.zeros((self.k, self.data.shape[1]))

    #计算距离一个样本点到类中心的距离
    def getDistances(self):
        #遍历数据集
        for i in range(self.data.shape[0]):
            distance_index = ((np.tile(self.data[i], (self.k, 1)) - self.centre) ** 2).sum(axis=1) ** 0.5
            #计算到每一个中心点的距离
            self.distances[i] = distance_index

    def getCetre(self):
        #生成随机数，每个样本点的初始化聚类
        self.classification = np.random.randint(0,self.k,(self.data.shape[0]))
        for i in range(self.k):
            self.classification[i]=i

    def classify(self):
        #对全部样本点都做批处理
        new_classification=np.argmin(self.distances,axis=1)
        #更新聚类
        if any(new_classification-self.classification):
            self.classification = new_classification
            return 1
        else:
            return 0

    #更新中心点
    def updateCentre(self):
        for i in range(self.k):
            #类内均值作为中心点
            self.centre[i] = np.mean(self.data[self.classification == i], axis=0)

    #聚类函数主体
    def main_cluster(self):
        #初始化，随即分类，获取中心
        self.getCetre()
        for i in range(self.maxiter):
            #更新中心点
            self.updateCentre()
            #求距离
            self.getDistances()
            #分类
            if (not self.classify()):
                break