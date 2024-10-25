# K-近邻算法是一种基本的分类与回归方法，他是基于数据集中的k个最近邻居的分类决策
from numpy import *
import operator
import matplotlib.pyplot as plt

# 从文件读入数据集
def file2matrix(filename) :
    file = open(filename)
    # 读取文件的每一行
    arrayOfLines = file.readlines()
    # 得到文件的行数
    numberOfLines = len(arrayOfLines)
    # 创建一个特征值的零矩阵
    returnMatrix = zeros((numberOfLines, 3))
    # 创建标签向量
    classLabelVector = []
    index = 0
    for line in arrayOfLines :
        # 去掉每行的回车符
        line = line.strip()
        # 将每行的属性值分割成列表
        listFromLine = line.split('\t')
        # 将特征值列表的每一个特征值存放到特征值矩阵的对应位置中
        returnMatrix[index, :] = listFromLine[0:3]
        # 将特征值标签存放到标签向量中，文件中标签在最后一列，使用负索引直接获取，并将其转换为整型量否则编译器会将标签视作字符串处理
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMatrix, classLabelVector

def autoNorm(dataSet) :
    # 获取每一列的最小值和最大值，也就是每一个特征的最小值和最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 计算每个特征的取值范围
    ranges = maxVals - minVals
    # 创建归一化矩阵
    normDataSet = zeros(shape(dataSet))
    # 获得数据集的行数、即大小
    m = dataSet.shape[0]
    # 将minVals复制m次，并将结果放入生成的大小匹配的二维数组中，作为归一化矩阵的每一列的最小值
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 除以ranges，得到归一化矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

"""
@brief kNN算法分类函数
@type  欧几里得距离
@param[in] inX 输入的向量，需要归一化
@param[in] dataSet 训练数据集，需要归一化
@param[in] labels 标签列表
@param[in] k 近邻数据的数量
@return 分类结果
"""
def kNN_Classify(inX, dataSet, labels, k):
    # 首先获取数据集的行数，即数据集的大小
    dataSize = dataSet.shape[0]

    ''' 以下为kNN算法求取欧氏距离的过程 '''
    # tile函数将输入数据inX复制dataSize次，并将结果放入生成的大小匹配的二维数组中，与数据集的每一个元素做差，得到数据集与输入数据的差异矩阵
    diffMat = tile(inX, (dataSize, 1)) - dataSet
    # 将差异矩阵中的每一个元素平方，得到平方差距矩阵
    sqDiffMat = diffMat ** 2
    # 对平方差距矩阵按行求和，得到每个样本与inX之间的平方距离，这是因为再该矩阵中每一行都是统一个样本的不同维度特征值与输入数据的差异的平方
    sqDistances = sqDiffMat.sum(axis=1)
    # 将上面得到的平方距离开根得到欧几里得距离(在这里获得了输入数据与数据集中每一个样本之间的相似度)
    distances = sqDistances**0.5
    ''' 以上为kNN算法求取欧式距离的过程，可以从以上的算法推及曼哈顿距离、切比雪夫距离等其他距离计算方式 '''

    # 对上述获得的输入数据与数据集中的样本之间的欧氏距离进行排序，并获得排序后的索引
    sortedDistIndicies = distances.argsort()
    # 构建一个用于计算在k个邻近样本中各类别出现的频率
    classCount = {}
    # 开始遍历前k个临近样本
    for i in range(k):
        # 将第i个最邻近样本点的标签取出
        voteIlabel = labels[sortedDistIndicies[i]]
        # 在字典classCount中将标签的出现频率++，如果标签不存在，则初始化为0
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)
    #再对得到的字典按照频率进行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 最后返回出现频率最高的标签作为分类结果
    return sortedClassCount[0][0]


def classifyPerson():
    # 创建标签列表
    resultList = ['not at all', 'in small doses', 'in large doses']
    # 获取用户输入的特征值
    percentTats = float(input(f"percentage of time spent playing video games?"))
    ffMiles = float(input(f"frequent flier miles earned per year?"))
    iceCream = float(input(f"ice cream size?"))
    # 从文件获取数据集和标准标签集
    datingData, datingLabels = file2matrix('datingTestSet2.txt')
    # 归一化数据集
    normMat, ranges, minVals = autoNorm(datingData)
    # 将用户输入的特征值转换为特征向量
    inArr = array([ffMiles, percentTats, iceCream])
    # 调用kNN算法分类，输入的参数也需要归一化
    classifierResult = kNN_Classify((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print(f"You will probably like this person: {resultList[classifierResult - 1]}")


if __name__ == "__main__":

    classifyPerson()