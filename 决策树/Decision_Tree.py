from math import log
from numpy import *
import matplotlib as plt


def calcShannonEnt(dataSet):
    """
    计算给定数据集的香农熵
    : param 数据集
    : return 香农熵
    """
    # 获得数据集中数据的数量
    numEntries = len(dataSet)
    # 创建空字典以存储标签
    labelCounts = {}
    # 从数据集中获取每个数据的特征向量
    for featVec in dataSet:
        # 假定特征向量的最后一个元素就是该数据的标签
        currentLabel = featVec[-1]
        # 如果该标签不在字典中，初始化该标签
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        # 标签出现的次数+1
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 将每个标签出现的次数除以数据总数得到该标签的数据在整个数据集中出现的概率
        prob = float(labelCounts[key]) / numEntries
        # 计算香农熵
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """
    划分数据集
    : dataSet 数据集
    : axis    划分数据集的轴，也就是特征的索引
    : value   给定划分特征
    """
    # 创建空返回列表
    retDataSet = []
    # 对于数据集中的每个数据：
    for featVec in dataSet:
        # 如果数据在划分轴上的特征与给定特征相同
        if featVec[axis] == value:
            # 则将该数据除了给定特征外的所有特征都取出并加入返回数据集中
            reducedFeatVec = featVec[:axis]
            # extend方法将一个列表中的元素添加到另一个列表中
            reducedFeatVec.extend(featVec[axis+1:])
            # append方法将元素添加到列表末尾，无论这个元素是列表还是其他类型
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    选择最优划分特征
    : dataSet 数据集
    : return 最优划分特征的索引
    """
    # 默认数据向量的最后一个元素为标签，因此特征数量为数据向量的数量减1
    numFeatures = len(dataSet[0]) - 1
    # 计算数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 获得所有特征的值
        featList = [example[i] for example in dataSet]
        # 利用集合转换去重
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 划分数据集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算划分后的数据集的香农熵
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        # 如果信息增益大于当前最优信息增益，则更新最优信息增益和最优特征
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


if __name__ == '__main__':
