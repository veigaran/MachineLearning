from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


# 导入数据
def file2matrix(filename):
    # 读取数据
    fr = open(filename)
    # numberLines为数据的行数
    numberLines = len(fr.readlines())
    # 生成矩阵，zeros(2,3)生成的为2*3的矩阵
    returnMat = zeros((numberLines, 3))
    # 用来保存对应的类别
    classLabelVector = []
    fr = open(filename)
    # 计数
    index = 0;
    for line in fr.readlines():
        # 移除字符串头尾指定字符
        line = line.strip()
        # 根据tab分割每一行数据
        listFromline = line.split('\t')
        # 往矩阵每一行添加数据
        returnMat[index, :] = listFromline[:3]
        classLabelVector.append(int(listFromline[-1]))
        index += 1
    return returnMat, classLabelVector


# 归一化
def autoNorm(dataSet):
    """归一化公式：
    Y = (X - Xci) / (Xmas - Xci)
    其中的 min和max
    分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """
    # 计算最小特征值和最大特征值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 计算极差
    ranges = maxVals - minVals
    # 生成原矩阵相同形状的矩阵，用于保存
    normDataset = zeros(shape(dataSet))
    # m为矩阵的行数
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    # tile(array, (m,n))函数作用是array在x轴复制n次，在y轴复制m次
    normDataset = dataSet - tile(minVals, (m, 1))
    normDataset = normDataset / tile(ranges, (m, 1))
    return normDataset, ranges, minVals


# kNN算法
def classify0(inX, dataSet, labels, k):
    '''
    :param inX: 用于分类的输入向量
    :param dataSet:输入的训练样本集
    :param labels:标签向量
    :param k:选择最近邻居的数目
    :return:
    '''
    # 1.计算距离--欧式距离
    dataSetsize = dataSet.shape[0]
    # 生成与训练样本对应的矩阵，并与训练样本求差
    '''  
        欧氏距离： 点到点之间的距离
           第一行： 同一个点 到 dataSet的第一个点的距离。
           第二行： 同一个点 到 dataSet的第二个点的距离。
           ...
           第N行： 同一个点 到 dataSet的第N个点的距离。
        [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
        (A1-A2)^2+(B1-B2)^2+(c1-c2)^2
    '''
    diffMat = tile(inX, (dataSetsize, 1)) - dataSet
    # 取平方
    sqDifMat = diffMat ** 2
    # 矩阵每一行相加
    sqDistance = sqDifMat.sum(axis=1)
    # 开方
    distance = sqDistance ** 0.5
    # 根据距离从小到大排序，并返回索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    # 例如：y=array([3,0,2,1,4,5]) 则，x[3]= 1最小，所以y[0]=3;x[5]=9最大，所以y[5]=5。
    sortedDis = distance.argsort()

    # 2.选择距离最小的几个点
    classCount = {}
    for i in range(k):
        # 找到样本的类型
        voteIlabel = labels[sortedDis[i]]
        # 在字典中将该类型加一，运用字典的get方法
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 3.排序并返回最多的那个类型
    # sorted 中的第2个参数 key=operator.itemgetter(1) 这个参数的意思是先比较第几个元素
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回字典中value最大的key
    return sortedClassCount[0][0]


# 测试算法：使用海伦提供的部分数据作为测试样本。如果预测分类与实际类别不同，则标记为一个错误。
def datingClassTest(filepath):
    # 设置测试数据的一个比例（训练数据集比例=1-hoRatio）
    # 即测试集和训练集
    hoRatio = 0.1
    # 加载数据
    datingDataMat, datingLabels = file2matrix(filepath)
    # 归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m即为行数
    m = normMat.shape[0]
    # numTestVecs表示测试的样本数量 ； m为训练的样本数量
    numTestVecs = int(m * hoRatio)
    print('numTestVecs=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


# 约会网站预测函数
def classifyPerson():
    resultList = ['不喜欢', '一般', '很喜欢']
    percentTats = float(input("玩视频游戏所耗时间百分比"))
    ffMiles = float(input("每年获得的飞行常客里程数"))
    iceCream = float(input("每周消费的冰淇淋公升数"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


if __name__ == '__main__':
    classifyPerson()
    # file = r'F:\A文档\python学习\MachineLearning\k-NearestNeighbor\2.KNN\datingTestSet2.txt'
    # datingClassTest(file)
    # datingDataMat, datingLabels = file2matrix(file)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # plt.show()
