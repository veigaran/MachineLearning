#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2020-05-03 16:24
@Author:Veigar
@File: DT_sklearn.py
@Github:https://github.com/veigaran
"""

import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import pydotplus

# 载入数据
adult_data = pd.read_csv('./DecisionTree.txt')
# 区分特征属性及目标
feature_columns = [u'workclass', u'education', u'marital-status', u'occupation', u'relationship', u'race', u'gender',
                   u'native-country']
label_column = ['income']
# 区分特征和目标列
features = adult_data[feature_columns]
label = adult_data[label_column]

# 特征工程；具体见pd.get_dummies用法，可理解为将属性特征用0、1表示
features = pd.get_dummies(features)
label = pd.get_dummies(label)

# 切分数据集
train_features, test_features, train_label, test_label = train_test_split(features, label, test_size=0.2)

# 初始化决策树分类器
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
# 拟合数据
clf.fit(train_features.values, train_label.values)

# 预测数据
predict = clf.predict(test_features.values)
print('预测完成')

print('回归树二乘偏差均值:', mean_squared_error(test_label.values, predict))
# 决策树可视化
# dot_data = tree.export_graphviz(clf, out_file=None, feature_names=features.columns, class_names=['<50k', '>50k'],
#                                 filled=True, rounded=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png('tree.png')
# # display(Image(graph.create_png()))
