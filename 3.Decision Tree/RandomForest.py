#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2020-05-03 19:50
@Author:Veigar
@File: RandomForest.py
@Github:https://github.com/veigaran
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

# 载入数据
boston_house = load_boston()
boston_feature_name = boston_house.feature_names
boston_features = boston_house.data
boston_target = boston_house.target

# 构建模型
rgs = RandomForestRegressor(n_estimators=15)
# 拟合数据
rgs = rgs.fit(boston_features, boston_target)
# 预测
rgs.predict(boston_features)
