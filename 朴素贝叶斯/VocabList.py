
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting','stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate','my','steak', 'how', 'to','stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food','stupid']]
    # 0代表正常言论，1代表侮辱性言论
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def createVocabList(dataSet):
    # 创建一个空集
    vocabSet = set([])
    for document in dataSet:
        # 创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
    将输入的单词集转换为向量
    :param vocabList 词汇表
    :param inputSet  某个文档
    """
    # 创建一个0向量，向量维数与词汇表的长度相同
    returnVec = [0] * len(vocabList)
    # 遍历文档中的每个单词，如果单词出现在词汇表中，则将对应位置的元素置1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print(f"the word: {word} is not in my Vocabulary!")
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    """
    构建贝叶斯词袋模型
    :param vocabList 词汇表
    :param inputSet  文档集
    """
    # 创建一个0向量，向量维度和词汇表的长度相同
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
