from math import log
import matplotlib.pyplot as plt
import operator

# 以下是将决策树可视化的函数
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def retrieveTree(i):
    """
    输出预先存储的树信息
    :param i 树的编号
    """
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},\
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt,\
                           textcoords='axes fraction', va="center", ha="center", bbox=nodeType,\
                           arrowprops=arrow_args)
    
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
# 以上是将决策树可视化的函数

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

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
        # 计算香农熵，这里以2为底求对数
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

def majorityCnt(classList):
    """
    获得出现次数最多的标签
    : classList 标签列表
    : return    出现次数最多的标签
    """
    # 创建标签空字典以键值形式存储标签及其出现次数
    classCount = {}
    # 遍历标签列表
    for vote in classList:
        if vote not in classCount.keys():
            # 如果标签不在字典中，初始化该标签
            classCount[vote] = 0
        # 标签出现的次数+1
        classCount[vote] += 1
    # 排序标签字典，并按照出现次数降序排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的标签
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    递归创建决策树
    : dataSet   数据集
    : labels    标签列表
    """
    # 假定数据集的最后一列向量中的元素为标签，通过索引获得标签列表
    classList = [example[-1] for example in dataSet]
    # 如果标签列表中所有元素都相同，则返回该标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集中只有一个数据，则返回出现次数最多的标签
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优划分数据
    bestFeature = chooseBestFeatureToSplit(dataSet)
    # 获得最优划分特征的标签
    bestFeatureLabel = labels[bestFeature]
    # 构建空字典以存储决策树
    myTree = {bestFeatureLabel: {}}
    # 删除最优划分特征的标签
    del(labels[bestFeature])
    # 获得最优划分
    featValues = [example[bestFeature] for example in dataSet]
    # 去重
    uniqueVals = set(featValues)
    # 遍历所有划分特征
    for value in uniqueVals:
        # 获得所有标签，注意这里已经删除了上一次划分得出的最有特征标签
        subLabels = labels[:]
        # 递归: 最优划分特征下的每一个标签都对应一个子树
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree

def getNumLeafs(myTree):
    """
    获得决策书的叶子节点数量
    : myTree 决策树
    : return 叶子节点数量
    """
    numLeafs = 0
    # 获得当前所在树(字典)的第一个键
    firstStr = list(myTree.keys())[0]
    # 获得当前键对应的值，也就是子树
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 遍历字数的键，如果键的类型是字典则递归调用，否则叶子节点数量+1
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    """
    获得决策树的深度
    : myTree 决策树
    : return 决策树的深度
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 深度的计算和叶子的计算十分相似，但区别为判为树枝的时候深度需要在当前树枝1的基础上递归计算
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def classify(inputTree, featLabels, testVec):
    """
    调用决策树算法进行分类，实际上就是测试向量遍历决策树的过程
    : inputTree   决策树
    : featLabels  特征标签列表
    : testVec     测试向量
    : return      分类结果
    """
    # 获得当前所在树的第一个键
    firstStr = list(inputTree.keys())[0]
    # 获得当前键对应的值，也就是子树
    secondDict = inputTree[firstStr]
    # 获得测试向量的第一个特征的索引
    featIndex = featLabels.index(firstStr)
    # 遍历子树的所有键
    for key in secondDict.keys():
        # 如果先前获得的特征索引在测试向量的元素与当前键相同，且当前键的类型是字典，则递归调用分类函数，否则返回当前键对应的值
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

if __name__ == '__main__':
    myDat, myLabels = createDataSet()
    # mytree = retrieveTree(1)
    myTree = createTree(myDat, myLabels)
    print("决策树：", myTree)
    createPlot(myTree)
