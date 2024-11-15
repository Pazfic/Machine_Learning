from numpy import *

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt', 'rb')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    """
    梯度上升算法
    :param  dataMatIn   输入的数据矩阵
    :param  classLabels 类别标签
    :return 权重
    """
    # 将输入的数据矩阵转换为numpy的mat类型
    dataMatrix = matrix(dataMatIn)
    # 将类别标签转换为列向量，以方便矩阵运算
    labelMat = matrix(classLabels).transpose()
    # 获取输入矩阵的形状，m为样本数(行数)，n为特征数(列数)
    m, n = shape(dataMatrix)
    # 设置步长
    alpha = 0.001
    # 设置最大迭代次数
    maxCycles = 500
    # 初始化权重矩阵为1矩阵，形状为(特征数，1)
    weights = ones((n, 1))
    for k in range(maxCycles):
        # 计算当前权重下的预测值h，这里dataMatrix * weights返回的是一个维数为100的列向量，所以h也是一个列向量
        h = sigmoid(dataMatrix * weights)
        # 计算预测值与实际标签的误差，返回的是列表(行向量)
        error = (labelMat - h)
        # 更新权重列向量
        weights = weights + alpha * dataMatrix.transpose() * error
    # 返回最终的权重矩阵
    return weights

def plotBestFit(weights):
    """
    画出决策边界
    :param weights 权重矩阵
    """
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y.transpose())
    plt.xlabel('X1')
    plt.ylabel('X2');
    plt.show()
    