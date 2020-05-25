#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2020-04-23 21:31
@Author:Veigar
@File: k-means.py
@Github:https://github.com/veigaran
"""
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# 文本向量化
def tf_idf_vector(corpus_path):
    """
    将文档向量化，注意此次为英文文档，中文文档应先进行分词
    :param corpus_path: 文件路径
    :return: 返回tf-idf向量，CountVectortizer的特征值
    """
    # 存放具体数据
    corpus_train = []
    # 遍历文件，按行读取
    for line in open(corpus_path):
        # 每一行根据”\t“切分
        line = line.strip().split('\t')
        if len(line) == 2:
            # 存放每一行的文本，并保存到列表
            words = line[1]
            corpus_train.append(words)
    print("build train-corpus done")
    # 利用CountVectorizer进行词频统计，初步转换，转换后格式为[1,0,2,1,0....],词袋模型
    count_v1 = CountVectorizer(max_df=0.4, min_df=0.01)
    counts_train = count_v1.fit_transform(corpus_train)

    # 存放词频统计后对应的单词
    word_dict = {}
    # count_V1_get_feature_names作用是获取词袋模型中的所有词语特征
    for index, word in enumerate(count_v1.get_feature_names()):
        word_dict[index] = word
    print("the shape of train is:")
    # repr()转为字符串
    print(repr(counts_train.shape))
    # tf-idf向量化
    tf_idf_transformer = TfidfTransformer()
    tf_idf_train = tf_idf_transformer.fit(counts_train).transform(counts_train)
    return tf_idf_train, word_dict


# k-means聚类函数
def cluster_k_means(tf_idf_train, word_dict, cluster_docs, cluster_keywords, num_clusters):
    """
    k-means聚类函数
    :param tf_idf_train: 向量矩阵，可以为tf-idf，也可以为one-hot等等
    :param word_dict:关键词字典
    :param cluster_docs:聚类结果保存文件
    :param cluster_keywords:聚类结果的关键词文件
    :param num_clusters:聚类族数
    :return:无
    """
    f_docs = open(cluster_docs, 'w+')
    # 生成k-means模型
    km = KMeans(n_clusters=num_clusters)
    # 拟合
    km.fit(tf_idf_train)
    # km.labels_：聚类标签，如生成五类，标签可以为[0,1,2,3,4],具体文件包含所有数据的标签如[3,0,1,4,4,1,2....]
    clusters = km.labels_.tolist()
    # 用于存储所有的标签，如｛0:1,1:2,...｝
    cluster_dict = {}
    # km.cluster_centers_：聚类中心
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    # 序号
    doc = 1
    for cluster in clusters:
        # 按行写入文件
        f_docs.write(str(str(doc)) + ',' + str(cluster) + '\n')
        doc += 1
        if cluster not in cluster_dict:
            cluster_dict[cluster] = 1
        else:
            cluster_dict[cluster] += 1
    f_docs.close()
    cluster = 1

    # 存储关键词文件
    f_cluster_words = open(cluster_keywords, 'w+')
    for ind in order_centroids:
        words = []
        # 每个聚类只保存50个词语
        for index in ind[:50]:
            words.append(word_dict[index])
        print(cluster, (','.join(words)))
        f_cluster_words.write(str(cluster) + '\t' + ','.join(words) + '\n')
        cluster += 1
        print('******' * 5)
    f_cluster_words.close()
    visualization(tf_idf_train, km)


# 数据可视化
def visualization(tf_idf_train, km):
    """
    将聚类结果进行展示
    :param tf_idf_train: 向量矩阵文件
    :param km: 拟合后的k-means模型
    :return: 作图
    """
    # 因输入的矩阵文件维度很高，需对权重进行降维处理，可选用TSNE算法或PCA算法

    pca = PCA(n_components=3)
    # 矩阵转为数组
    tf_idf_weight = tf_idf_train.toarray()

    '''三维展示'''
    embedded = pca.fit_transform(tf_idf_weight)
    # 对数据进行归一化操作
    x_min, x_max = np.min(embedded, 0), np.max(embedded, 0)
    embedded = embedded / (x_max - x_min)

    # 创建显示的figure
    fig = plt.figure()
    ax = Axes3D(fig)
    # 将数据对应坐标输入到figure中，不同标签取不同的颜色
    ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2], c=km.labels_)

    # 关闭了plot的坐标显示
    plt.axis('off')
    plt.show()
    '''二维展示'''
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(tf_idf_weight)
    x = []
    y = []

    # 将降维后的数据对应x、y坐标
    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])
    # 画布大小
    fig = plt.figure(figsize=(10, 10))
    # 设定图形区
    # ax = plt.axes()
    # x轴名称
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')

    # 设置标题名称
    plt.title('K-means clustering')
    # 设置散点图不同种类不同颜色
    cValue = ['r', 'y', 'g', 'b', 'r', 'y']
    # c参数为色彩或者颜色序列，marker为显示方式，有x，o等等
    plt.scatter(x, y, c=km.labels_, marker=".")
    # 设置刻度标签 xticks 字体大小
    plt.xticks(())
    plt.yticks(())
    # 展示
    plt.show()
    # plt.savefig('./sample.png', aspect=1)



# 族数对分类的影响分析
def best_kmeans(tfidf_matrix, word_dict):
    K = range(1, 10)
    meandistortions = []
    for k in K:
        print(k), ('****' * 5)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(tfidf_matrix)
        meandistortions.append(
            sum(np.min(cdist(tfidf_matrix.toarray(), kmeans.cluster_centers_, 'euclidean'), axis=1)) /
            tfidf_matrix.shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for K-means clustering')
    plt.show()


def main():
    corpus_train = r'./corpus_train.txt'
    cluster_docs = r'./cluster_result_document.txt'
    cluster_keywords = r"cluster_result_keyword.txt"
    num_clusters = 6
    tf_idf_train, word_dict = tf_idf_vector(corpus_path=corpus_train)
    # best_kmeans(tf_idf_train, word_dict)
    cluster_k_means(tf_idf_train, word_dict, cluster_docs, cluster_keywords, num_clusters)


if __name__ == '__main__':
    main()
