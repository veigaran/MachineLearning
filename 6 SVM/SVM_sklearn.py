"""
1、文本向量化
2、构建分类模型
3、模型测试及预测
"""
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics


def readData(file):
    data = pd.read_csv(file, header=None, sep='\t')
    dataMat = [i for i in data[1]]
    labelMat = [i for i in data[0]]
    print(len(dataMat), dataMat[0:10])
    train = [dataMat, labelMat]
    return train


def cutWords(data_list):
    cutString = []
    for i in data_list:
        a = list(jieba.cut(i))
        b = "".join(a)
        cutString.append(b)
    return cutString

    # 模型预测


def model_predict(text, model, tf):
    """
    :param text: 单个文本
    :param model: 朴素贝叶斯模型
    :param tf: 向量器
    :return: 返回预测概率和预测类别
    """
    text1 = [" ".join(jieba.cut(text))]
    # 进行tfidf特征抽取
    text2 = tf.transform(text1)
    predict_type = model.predict(text2)[0]
    return predict_type


def svm(trainFile, testFile):
    train = readData(trainFile)
    test = readData(testFile)

    train_dataMat = cutWords(train[0])
    train_labelMat = train[1]
    test_dataMat = cutWords(test[0])
    test_labelMat = test[1]

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2))
    train_features = tfidf.fit_transform(train_dataMat)
    print(train_features.shape)

    test_features = tfidf.transform(test_dataMat)
    print(test_features.shape)

    clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                    intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                    multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                    verbose=0)

    # 模型训练
    clf.fit(train_features, train_labelMat)
    print(clf.score(train_features, train_labelMat))
    print('训练集准确率：', metrics.accuracy_score(test_labelMat, clf.predict(test_features)))
    print('训练集F1值：', metrics.f1_score(test_labelMat, clf.predict(test_features), average='micro'))
    return clf, tfidf


if __name__ == '__main__':
    trainFile = r'./data/dev.tsv'
    testFile = r'./data/test.tsv'
    clf, tfidf = svm(trainFile, testFile)
    print('开始测试')
    s = 'Method I studied the evolution of the number of articles and citations in journals indexed in the Science Citation Index (SCI) and in the Social Sciences Citation Index (SSCI) from 1998 to 2007 {ref[#bib 7]} , using data published in the Journal Citation Reports (JCR) database and included in the Web interface available to Spanish universities . All journals with data on articles and citations greater than zero were included . To obtain ranks of the number of articles and citations , all journals were listed from highest to lowest using Excel pivot tables to detect and remove duplicates in the values of articles or citations . Next , rank order was assigned from 1 to N. The plot of ranks suggested that they did not fit a typical power law because of the bending tail on the right-hand side of the curve (see Figures 1 -- 4) . Instead , I used the function suggested by Mansilla et al. (2007 {ref[#bib 7]}) . For every year , I calculated the value of the parameters with the linear least squares method . Previously , I transformed the equations using logarithms to yield : where Y = Citations or articles , r = Rank , N = Maximum rank . The method was similar to the one used by Mansilla et al. and in the previous study (Campanario , in press) . The equation suggested by Mansilla et al. (2007 {ref[#bib 7]}) has some advantages . For example , the parameter a is more influential for small values of r . When r is low , the law become nearly Lotkaian . However , as r increases , the influence of parameter b also rises . This combination of influences can explain the decrease in values as the abscissa increases (Mansilla et al. , 2007 {ref[#bib 7]}) . All calculations were done with the online service hosted at www.zunzun.com Once the parameters K , b , and a were found , the theoretical curve was obtained and plotted . '
    predict_type = model_predict(s, clf, tfidf)
    print(predict_type)
