'''
参考：https://github.com/apachecn/AiLearning/blob/master/docs/ml/3.%E5%86%B3%E7%AD%96%E6%A0%91.md
'''
from __future__ import print_function

print(__doc__)
from math import log
import operator
import decisionTreePlot as dtPlot


# 基础数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 计算给定数据集的香农熵
def calcShannonEnt(dataset):
    '''
    求香农熵，具体见百度
    :param dataset:输入数据集
    :return:香农熵
    '''
    # 数据集长度
    numEntries = len(dataset)
    # 用于保存标签
    labelCounts = {}
    # 数据集中每个元素遍历，若不存在则添加到labelCounts中
    for featVec in dataset:
        # 每行数据最后一个元素为标签
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 计数
        labelCounts[currentLabel] += 1

    # 香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 求概率公式
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
        # print(labelCounts[key])
    return shannonEnt


# 按照给定特征划分数据集
# 将指定特征的特征值等于 value 的行剩下列作为子数据集。
def splitDataSet(dataSet, index, value):
    '''
    该函数是依据index列进行分类，若index列的数据等于value的值，就把index划分到我们创建的新的数据集中
    :param dataSet:输入的待划分数据集
    :param index:表示每一行的index值， 划分数据集的特征
    :param value:表示index列对应的value值  需要返回的特征的值
    :return:index列为value的数据集【该数据集需要排除index列】
    简而言之，提取数据集中，每一行index列对应的value值，且该value与给定的value相等，同时除去index列的一个新的数据集

    '''
    # 用于保存数据
    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            # 除去index列对应的数据
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    '''
    :param dataSet: 数据集
    :return:最优的特征列
    '''
    # 求第一行有多少列的Feature，最后一列为label列，所以减一
    numFeatures = len(dataSet[0]) - 1
    # 数据集的原始信息熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最优的信息增益值、最优的Feature编号
    bestInfoGain, bestFeature = 0.0, -1
    # 每一行中的元素遍历
    for i in range(numFeatures):
        # 获取数据集第i列的所有数据
        featList = [example[i] for example in dataSet]
        # 对数据去重
        uniqueVals = set(featList)
        # 设定新信息熵
        newEntropy = 0.0
        # 遍历第i列组成的value集合
        # 计算该列的信息熵
        for value in uniqueVals:
            # 数据划分
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算概率
            prob = len(subDataSet) / float(len(dataSet))
            # 计算信息熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益：原始信息熵与新信息熵之差
        infoGain = baseEntropy - newEntropy
        print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
        # 若该信息增益大于原始信息增益，则更换，并返回该列的索引值，即第i列
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 选择次数出现最多的一个结果
def majorityCnt(classList):
    '''
    :param classList: label列组成的集合
    :return:最优的特征列，即次数最多的结果
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建树，运用了迭代
def createTree(dataSet, labels):
    # 选择数据集中的label
    classList = [example[-1] for example in dataSet]

    # 若数据集的最后一列的第一个值出现的次数=整个集合的数量，也即只有一个类别，直接返回结果
    # 第一个停止条件：所有的类别标签都相同，则直接返回该标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 若数据集只有1列，那么最初出现label次数做多的一类即为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分为仅包含唯一类别的分组
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 主函数
    # 选择最好的数据集划分的方式对应的索引值
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 选择最好的标签
    bestFeatLabel = labels[bestFeat]
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del (labels[bestFeat])
    # 数据集最优列，用它做branch分类
    featValues = [example[bestFeat] for example in dataSet]
    # 去重
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签
        subLabels = labels[:]
        # 迭代 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 给输入的节点进行分类
def classify(inputTree, featLabel, testVec):
    '''
    :param inputTree:决策树模型
    :param featLabel:feature对应的名称
    :param testVec:测试输入的数据
    :return:分类的结果值
    '''
    # 获取tree的根节点对应的key值
    firstStr = list(inputTree.keys())[0]
    # 根节点对应的value值
    secondDict = inputTree[firstStr]
    # 获取根节点在label中的索引值，从而确定先后顺序，从而确定输入的testVec怎么开始对照树来做分类
    featIndex = featLabel.index(firstStr)
    # 找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]

    # ？？？？
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    if (isinstance(valueOfFeat, dict)):
        classLabel = classify(valueOfFeat, featLabel, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def get_tree_height(tree):
    """
     Desc:
        递归获得决策树的高度
    Args:
        tree
    Returns:
        树高
    """

    if not isinstance(tree, dict):
        return 1

    child_trees = list(tree.values())[0].values()

    # 遍历子树, 获得子树的最大高度
    max_height = 0
    for child_tree in child_trees:
        child_tree_height = get_tree_height(child_tree)

        if child_tree_height > max_height:
            max_height = child_tree_height

    return max_height + 1


def fishTest():
    # 1.创建数据和结果标签
    myDat, labels = createDataSet()
    # print myDat, labels

    # 计算label分类标签的香农熵
    # calcShannonEnt(myDat)

    # # 求第0列 为 1/0的列的数据集【排除第0列】
    # print '1---', splitDataSet(myDat, 0, 1)
    # print '0---', splitDataSet(myDat, 0, 0)

    # # 计算最好的信息增益的列
    # print chooseBestFeatureToSplit(myDat)

    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print(classify(myTree, labels, [1, 1]))

    # 获得树的高度
    print(get_tree_height(myTree))

    # 画图可视化展现
    dtPlot.createPlot(myTree)


if __name__ == '__main__':
    fishTest()
