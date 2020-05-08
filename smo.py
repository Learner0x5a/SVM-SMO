'''
参考源码：https://github.com/huxinliang8888/svm
SVM SMO算法详细推导及总结：https://blog.csdn.net/weixin_42001089/article/details/83276714

重构代码，封装，实现类似于sklearn的接口
'''
import numpy as np
class BinarySVC:
    def __init__(self,C,toler,maxIter,**kernelargs):
        self.C = C # 对分类错误的惩罚程度；可以尝试：两类点采用两种C，从而实现对误报/漏报的偏向
        self.toler = toler # 误差限
        self.maxIter = maxIter # 迭代限
        self.kwargs = kernelargs # 核函数的参数

        self.x = None
        self.label = None
        self.m = None
        self.alpha = None
        self.b = None
        self.E = None
        self.K = None

        self.support_vector_x = None # 支持向量
        self.support_vector_index = None
        self.support_vector_alpha = None
        self.support_vector_label = None
        self.w = 0.

    # 计算误差
    # 可以尝试：调高欺诈数据的loss权重，就是当label=欺诈时，给原始的E乘个系数
    def EK(self,k):
        fxk = np.dot(self.alpha*self.label,self.K[:,k])+self.b
        Ek = fxk - float(self.label[k])
        return Ek
    # 更新误差
    def updateEK(self,k):
        Ek = self.EK(k)
        self.E[k] = [1 ,Ek] # 更新过误差的，第一个元素为1

    # 随机选择第2个alpha
    def selectJrand(self,i,m):
        j = i
        while j == i:
            j = int(np.random.uniform(0,m))
        return j

    # 调整大于H或小于L的alpha值
    def clipAlpha(self,a_j,H,L):
        if a_j > H:
            a_j = H
        if L > a_j:
            a_j = L
        return a_j

    def selectJ(self,i,Ei): # 找最大化|E1-E2|的j
        maxE = 0.
        selectJ = 0
        Ej = 0.
        
        validECacheList = np.nonzero(self.E[:,0])[0] # 已经更新过误差的x
        if len(validECacheList) > 1:
            for k in validECacheList: # 找最大化|E1-E2|的j
                if k == i:
                    continue
                Ek = self.EK(k)
                deltaE = abs(Ei-Ek)
                if deltaE > maxE:
                    selectJ = k
                    maxE = deltaE
                    Ej = Ek
            return selectJ,Ej
        else: # 最初随机选一个j
            selectJ = self.selectJrand(i,self.m)
            Ej = self.EK(selectJ)
            return selectJ,Ej

    def innerL(self, i): # 主功能
        Ei = self.EK(i)
        # 找第一个alphas[i]，即不满足KKT条件的alpha
        if (self.label[i] * Ei < -self.toler and self.alpha[i] < self.C) or (self.label[i] * Ei > self.toler and self.alpha[i] > 0): 
            self.updateEK(i)
            j,Ej = self.selectJ(i,Ei) # 找第2个alphas[j]，启发式(找最大化|E1-E2|的j)

            # 保存上一步的alpha
            alphaIOld = self.alpha[i].copy()
            alphaJOld = self.alpha[j].copy()

            # 计算修正alpha所需的边界值H/L
            if self.label[i] != self.label[j]: 
                L = max(0,self.alpha[j]-self.alpha[i])
                H = min(self.C,self.C + self.alpha[j]-self.alpha[i])
            else:
                L = max(0,self.alpha[j]+self.alpha[i] - self.C)
                H = min(self.C,self.alpha[i]+self.alpha[j])
            if L == H:
                return 0

            # 计算eta
            eta = 2*self.K[i,j] - self.K[i,i] - self.K[j,j]
            if eta >= 0:
                return 0
            # 计算新的alpha2，不考虑约束
            self.alpha[j] -= self.label[j]*(Ei-Ej)/eta
            # 修正alpha2
            self.alpha[j] = self.clipAlpha(self.alpha[j],H,L)
            # 更新误差E2，误差由一个矩阵维护
            self.updateEK(j)
            # 如果误差E2变化很小，退出
            if abs(alphaJOld-self.alpha[j]) < 0.00001:
                return 0
            # 更新alpha1
            self.alpha[i] +=  self.label[i]*self.label[j]*(alphaJOld-self.alpha[j])
            # 更新误差E1
            self.updateEK(i)

            # 计算新的b
            b1 = self.b - Ei - self.label[i] * self.K[i, i] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[i, j] * (self.alpha[j] - alphaJOld)
            b2 = self.b - Ej - self.label[i] * self.K[i, j] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[j, j] * (self.alpha[j] - alphaJOld)
            # b的更新规则
            if 0 < self.alpha[i] and self.alpha[i] < self.C:
                self.b = b1
            elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) /2.0

            return 1
        else:
            return 0

    def fit(self,dataMat,classlabels): 
        # 初始化
        self.x = np.array(dataMat)
        self.label = classlabels
        self.m = np.shape(dataMat)[0]
        self.alpha = np.array(np.zeros(self.m),dtype='float64') # 初始化alpha
        self.b = 0. # 初始化b
        self.E = np.array(np.zeros((self.m,2))) # 维护一个误差矩阵，每行对应一个x，每行第一个元素标记是否更新过误差，第二个元素为误差值
        self.K = np.zeros((self.m,self.m),dtype='float64') # 初始化內积矩阵
        # 应用核函数
        for i in range(self.m):
            for j in range(self.m):
                self.K[i,j] = self.Kernel(self.x[i,:],self.x[j,:])

        # 主循环
        iter = 0
        entireSet = True # 全集遍历的标识
        alphaPairChanged = 0
        while iter < self.maxIter and ((alphaPairChanged > 0) or (entireSet)):
            alphaPairChanged = 0
            if entireSet: # 全集遍历
                for i in range(self.m):
                    alphaPairChanged += self.innerL(i)
                iter += 1
            else: # 非边界遍历
                nonBounds = np.nonzero((self.alpha > 0)*(self.alpha < self.C))[0]
                for i in nonBounds:
                    alphaPairChanged += self.innerL(i) 
                iter += 1
            
            # 全集/非全集交替进行
            if entireSet:
                entireSet = False
            elif alphaPairChanged == 0: # 如果非边界的点没有更新alpha，切换回全集遍历
                entireSet = True
        
        # 迭代结束，整理系数，计算w
        self.support_vector_index = np.nonzero(self.alpha)[0] # 非0的alpha的索引，即支持向量
        self.support_vector_x = self.x[self.support_vector_index] # 非0的alpha对应的x
        self.support_vector_alpha = self.alpha[self.support_vector_index] # 非0的alpha
        self.support_vector_label = self.label[self.support_vector_index] # 非0的alpha对应的y
        self.calcw()

        # 清理内存
        self.x = None
        self.K = None
        self.label = None
        self.alpha = None
        self.E = None

    def Kernel(self,x,z): # 两个向量的核函数
        if np.array(x).ndim != 1 or np.array(z).ndim != 1:
            raise Exception("input vector is not 1 dim")
        if self.kwargs['kernel'] == 'linear':
            return np.sum(x*z)
        elif self.kwargs['kernel'] == 'rbf':
            theta = self.kwargs['theta']
            return np.exp(np.sum((x-z)*(x-z))/(-1*theta**2))

    def calcw(self): # 计算w
        for i in range(self.m):
            self.w += np.dot(self.alpha[i]*self.label[i],self.x[i,:])

    def transform(self,testData):
        test = np.array(testData)
        result = []
        for i in range(np.shape(test)[0]): # 对每个测试样本计算预测值
            pred = self.b
            for j in range(len(self.support_vector_index)): # 计算f(x)
                pred += self.support_vector_alpha[j] * self.support_vector_label[j] * self.Kernel(self.support_vector_x[j],test[i,:])
            result.append(pred)
        result = np.asarray(result)
        # result = result/max(abs(result)) # 归一化
        # 分类阈值
        # result = (result > 0.).astype(np.int)
        return result
