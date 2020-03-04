"""
朴素贝叶斯分类器
基于贝努利模型，即不考虑每个单词出现的次数
"""
from numpy import *


def loadDataSet():
    """
    相当于训练集
    :return:
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # [0,0,1,1,1......]
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return posting_list, classVec


def createVocabList(dataset):
    """
    获得所有单词
    :param dataset:数据集
    :return: 去重后的单词列表
    """
    vocabList = set([])
    for document in dataset:
        # |表示并集
        vocabList = vocabList | set(document)
    return vocabList


def setWord2Vec(vocabList, inputSet):
    """
    将输入的inputset由词转为向量
    :param vocabList: 已有的单词列表
    :param inputSet: 输入的数据
    :return:
    """
    # 构造一个全是0的长度和词汇表长度相等的向量
    returnVec = [0] * len(vocabList)  # [0,0,0...]
    for word in inputSet:
        # 若该单词在词汇表中，则将对应的文档向量设为1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('Not in vocabulary ')
    return returnVec


def NB(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类训练器
    :param trainMatrix:输入的文件单词矩阵 如[[1,0,0,0,1,0....],[0,0,1....],...]
    :param trainCategory:文件对应的类别 如[1,0,0,1,...] 其中1表示侮辱性，0表示无侮辱性
    :return:
    """
    # 文档数量
    numTrainDocs = len(trainMatrix)
    # 每个文档的单词数量
    numWords = len(trainMatrix[0])
    # 侮辱性文档的概率，即用侮辱标记的文档数与总文档数之比
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 构造单词出现的次数列表，其中为了避免分子为0，故使用ones
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    # 整个数据集出现的单词数；同理，为了避免分子分母为0
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 若为侮辱性文档，则对侮辱性文件的向量进行相加
            p1Num += trainMatrix[i]  # [0,1,1,0,1...]+[1,0,1,0,0..]=[1,1,2,0,1.. ]
            # 统计侮辱性文档中单词的总数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    # 即 在1类别下，每个单词出现的概率
    p1Vect = log(p1Num / p1Denom)
    # 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    p0Vect = log(p0Num / p0Denom)
    return p1Vect, p0Vect, pAbusive


def classify(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    分类函数
     # 将乘法转换为加法
        乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
        加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    :param vec2Classify: 待测数据，形式为[0,0,1,0,1...]的向量形式
    :param p0Vec:类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    :param p1Vec:类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    :param pClass1:侮辱性文档出现的概率
    :return:
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    # 比较二者的概率大小
    if p1 > p0:
        return 1
    else:
        return 0


def test():
    # 载入数据，获得训练数据集，样本类别
    listPosts, listClass = loadDataSet()
    # 获得词汇表
    myVocabList = createVocabList(listPosts)
    # 创建数据矩阵
    trainMat = []
    for pos in listPosts:
        trainMat.append(setWord2Vec(myVocabList, pos))
    # 训练数据
    p0V, p1V, pAb = NB(array(trainMat), array(listClass))
    # 测试数据
    testlist = ['abb', 'dbb']
    thisDoc = array(setWord2Vec(myVocabList, testlist))
    print(testlist, classify(thisDoc, p0V, p1V, pAb))


def cutWord(string):
    import re
    listOfTokens = re.split(r'\W*', string)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = cutWord(open('./data/spam/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(1)
        wordList = cutWord(open('./data/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del (trainSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainSet:
        trainMat.append(setWord2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V, p1V, pSpam = NB(array(trainMat), array(trainClass))
    errorRate = 0.0
    for docIndex in testSet:
        wordVector = setWord2Vec(vocabList, docList[docIndex])
        if classify(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorRate += 1
    print('the errorCount is: ', errorRate)
    print('the testSet length is :', len(testSet))
    print('the error rate is :', float(errorRate) / len(testSet))
