from math import log
import operator
import matplotlib.pyplot as plt


decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="sawtooth", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', \
                            xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType,\
                            arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.axl = plt.subplot(111, frameon=False)
    plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('叶子节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 0, 'no'],
               [1, 1, 'maybe']]
    labels = ['no surfacing', 'filppers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCount = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCount.keys() :
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """
    划分数据集
    :param dataSet: 待划分的数据集
    :param axis:    划分数据集的特征
    :param value:   需要返回的特征的值
    :return:        划分后的数据集
    """
    # 创建空列表用于存储划分后的数据集
    retDataSet = []
    for featVec in dataSet:
        # 如果该样本的特征值与需要返回的特征值相同，则将该样本抽取出来划分到新的数据集中
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def majorityCnt(classList):
    """
    计算多数表决
    :param classList: 划分数据集后的类别列表
    :return:          多数表决结果
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def chooseBestFeatureToSplit(dataSet):
    """
    选择最优特征进行划分
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def createTree(dataSet, labels):
    """
    递归构建决策树
    :param dataSet: 数据集
    :param labels:  标签集，算法本身不需要这个变量，但是为了给出数据明确的含义，需要给出标签机
    :return:
    """
    # 获得数据集的所有特征标签
    classList = [example[-1] for example in dataSet]
    # 停止条件1: 如果数据集中所有实例属于同一类别，直接返回该类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 停止条件2: 如果使用完了所有特征仍不能将数据集划分成仅包含唯一类别的分组，则返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 获得当前数据集的最佳划分特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 创建空嵌套字典用于存储子树
    myTree = {bestFeatLabel: {}}
    # 去除标签列表中的最佳特征
    subLabels = labels.copy()
    del labels[bestFeat]
    # 获得数据集的所有特征标签
    featValues = [example[bestFeat] for example in dataSet]
    # 去除重复的特征值
    uniqueVals = set(featValues)
    # 遍历特征值集合中的每个特征
    for value in uniqueVals:
        # 递归构建子树: 最优划分特征下的每个值都对应一个子树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    # 返回子树
    return myTree

def getNumLeafs(myTree):
    """
    计算叶子节点数
    :param myTree: 生成的决策树
    :return:       叶子节点数
    """
    numLeafs = 0
    firstStr = myTree.Keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def getTreeDepth(myTree):
    """
    计算决策树的深度
    :param myTree: 生成的决策树
    :return:       决策树的深度
    """
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        lese: thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    """
    根据索引值获取决策树
    :param i:
    :return:
    """
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}, {'no surfaceing': {0 : 'no', 1: {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]

if __name__ == "__main__":
    dataSet, labels = createDataSet()
    myTree = createTree(dataSet, labels)

