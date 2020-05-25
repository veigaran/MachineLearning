#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2020-04-19 16:43
@Author:Veigar
@File: Decision_tree.py
@Github:https://github.com/veigaran
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# 载入数据集
boston = load_boston()
print(boston.feature_names)

# 获取特征集、房价
features = boston.data
prices = boston.target
# 数据集切分
train_features, test_features, train_price, test_price = train_test_split(features, prices, test_size=0.33)

# 构造CART回归树
dtr = DecisionTreeRegressor()
# 模型拟合
dtr.fit(train_features, train_price)
# 数据预测
predict_price = dtr.predict(test_features)

# 结果评价
print(mean_squared_error(test_price, predict_price))  # 回归树二乘偏差均值
