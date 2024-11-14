from VocabList import *
from numpy import *

def trainNB0(trainMatrix, trainCategory):
    """
    训练朴素贝叶斯分类器
    :param trainMatrix   训练文档矩阵
    :param trainCategory 训练文档类别标签向量
    """
    # 获得文档的数目和词汇的数目，这里假设所有文档的词汇数目相同
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 计算先验概率，计算文档属于类型1的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs) 
    # 初始化条件概率，p0num和p1num分别表示属于类型0和类型1的词汇出现的次数
    p0num = ones(numWords)
    p1num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 如果文档属于与类型1，则p1Num和p1Denom分别加上文档中词汇出现的次数和文档的总词汇数，否则对p0num和p0Denom作同样的操作
        if trainCategory[i] == 1:
            p1num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 计算条件概率，p1Vect和p0Vect分别表示条件概率向量，计算得到在类别1和类别0下词汇出现的概率
    p1Vect = log(p1num / p1Denom)
    p0Vect = log(p0num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类函数
    :param vec2Classify 待分类的文档向量
    :param p0Vec        类别0下词汇出现的概率向量
    :param p1Vec        类别1下词汇出现的概率向量
    :param pClass1      类别1的先验概率
    :return 文档属于类别1或者类别0
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(f'{testEntry} classified as: {classifyNB(thisDoc, p0V, p1V, pAb)}')
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(f'{testEntry} classified as: {classifyNB(thisDoc, p0V, p1V, pAb)}')

def textParse(bigString):
    """
    解析文本，将文本分割为词列表
    """
    import re
    # 利用正则表达式分割字符串、去除标点符号、转化为小写
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 因为文本是英文的，所以这里使用ISO-8859-1编码读取文本文件并解析
        wordList = textParse(open('/home/pazfic/Git_ws/Machine_Learning/朴素贝叶斯/email/spam/%d.txt' % i,\
                                  encoding='ISO-8859-1').read())
        # 将词列表作为元素添加到文档列表中
        docList.append(wordList)
        # 将词列表中的所有词汇添加到fullText列表中
        fullText.extend(wordList)
        # 将类别1添加到类别列表中
        classList.append(1)
        wordList = textParse(open('/home/pazfic/Git_ws/Machine_Learning/朴素贝叶斯/email/ham/%d.txt' % i,\
                                  encoding='ISO-8859-1').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 由文档列表创建词汇列表
    vocabList = createVocabList(docList)
    # 创建训练集
    trainingSet = list(range(50))
    testSet = []
    # 随机选取10个作为测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    # 建立训练矩阵和类别标签向量
    trainMat = []
    trainClasses = []
    # 遍历训练集，将文档向量添加到训练矩阵中
    for docIndex in trainingSet:
        
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errorCount += 1
    print(f'the error rate is {float(errorCount) / len(testSet)}')

if __name__ == '__main__':
    spamTest()
