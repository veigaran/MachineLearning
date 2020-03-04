"""
逻辑回归分类算法
"""
from numpy import *
import matplotlib.pyplot as plt


# 载入数据
def loadData(file_name):
    """
    :param file_name:文件
    :return: 返回数据矩阵、标签矩阵
    """
    dataMat = []
    labelMat = []
    fr = open(file_name)
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 该数据集第一列、第二列为数据，第三列为标签
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# sigmoid越阶函数
def sigmoid(inX):
    return 2 * 1.0 / (1 + exp(-2 * inX) - 1)


# Logistic回归梯度上升优化算法
def gradAscent(dataMat, classLabels):
    """
    :param dataMat: 数据数组，每列表示不同的特征
    :param classLabels: 标签数组
    :return:
    """
    # 转为矩阵
    dataMatrix = mat(dataMat)
    # 为方便计算，将其转为列向量
    labelMat = mat(classLabels).transpose()
    # m为行数也即样本量，n为列数，也即特征量
    m, n = shape(dataMatrix)
    # 代表每步移动的步长
    alpha = 0.001
    # 最大迭代次数
    maxCycles = 500
    # 生成一个和特征数目相同的向量，也即初始的回归系数向量w
    weights = ones((n, 1))
    # 参考http://blog.csdn.net/achuo/article/details/51160101
    for k in range(maxCycles):
        # 矩阵乘法
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return array(weights)


# 随机梯度上升算法
def stocGradAscent(dataMatrix, classLabels):
    """
    因上一种算法需遍历整个数据集，计算量大，因而每次随机选取部分数据集进行拟合
    :param dataMatrix:
    :param classLabels:
    :return:
    """
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for k in range(m):
        #  # sum(dataMatrix[i]*weights)为了求 f(x)的值，
        #  f(x)=a1*x1+b2*x2+..+nn*xn,此处求出的 h 是一个具体的数值，而不是一个矩阵
        h = sigmoid(sum(dataMatrix[k] * weights))
        # 为向量，计算真实类别与预测类别之间的差值，然后按照该差值调整回归系数
        error = classLabels[k] - h
        weights = weights + alpha * dataMatrix[k] * error
    return weights


# 随机梯度上升算法-优化版
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    上一个算法中，不同参数x1、x2...达到稳定值的迭代次数差异很大，为解决此问题进行优化
    :param dataMatrix:
    :param classLabels:
    :param numIter:随机梯度，也即迭代次数
    :return:
    """
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        #  [0, 1, 2 .. m-1]
        dataIndex = range(m)
        for i in range(m):
            # 设置alpha的值为随机值，会不断减少，但不会为0
            alpha = 4 / (1.0 + j + i) + 0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))
            error = classLabels[dataIndex[randIndex]] - h
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
    return weights


# 画图
def plotBestFit(dataArr, labelMat, weights):
    """
    :param dataArr: 样本数据的特征
    :param labelMat: 标签矩阵
    :param weights: 回归系数
    :return:
    """
    n = shape(dataArr)[0]
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[0]) == 1:
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
    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def test():
    # 1.收集数据
    dataMat, labelMat = loadData('./data/TestSet.txt')
    # 2.训练数据
    dataArr = array(dataMat)
    weights = gradAscent(dataMat, labelMat)
    # 3.数据可视化
    plotBestFit(dataArr, labelMat, weights)


# -----------------分隔符--------------------------#
def classifyVector(inx, weights):
    prob = sigmoid(sum(inx * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frtrain = open('./data/train.txt')
    frtest = open('./data/test.txt')
    trainSet = []
    trainLabels = []
    for line in frtrain.readlines():
        curline = line.strip().split()
        lineArr = []
        for i in range(21):
            lineArr.append(float(curline[i]))
        trainSet.append(lineArr)
        trainLabels.append(float(curline[21]))
    trainWeights = stocGradAscent1(array(trainSet), trainLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    # 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率
    for line in frtest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


# 调用 colicTest() 10次并求结果的平均值
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
    test()
