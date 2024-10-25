from os import listdir
from numpy import *
import operator
import matplotlib.pyplot as plt

def kNN_Classify(inputdata, dataSet, labels, k):
    dataSize = dataSet.shape[0]

    ''' 求欧氏距离 '''
    diffMat = tile(inputdata, (dataSize, 1)) - dataSet
    squaredDiffMat = diffMat ** 2
    squareDistance = squaredDiffMat.sum(axis=1)
    distances = squareDistance ** 0.5

    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistances[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


"""
@brief 将图像转换为特征向量的函数
@param filename 图像文件名，图像需要与脚本程序在统一目录下
"""
def img2Vector(filename):
    returnVector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0, 32 * i + j] = int(lineStr[j])
    return returnVector

def handwritingClassTest():
    # 创建标签列表
    hwLabels = []
    # 从trainingDigits文件夹中导入训练数据集
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2Vector('trainingDigits/%s' %fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector('testDigits/%s' %fileNameStr)
        classifierResult = kNN_Classify(vectorUnderTest, trainingMat, hwLabels, 5)
        print(f"the classifier came back with: %d, the real answer is: %d" %(classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print(f"\nthe total number of errors is: %d" %errorCount)
    print(f"\nthe total error rate is: %f" %(errorCount/float(mTest)))

if __name__ == '__main__':
    handwritingClassTest()
