#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2020-05-06 18:04
@Author:Veigar
@File: GBDT_Regression.py
@Github:https://github.com/veigaran
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# 数据读取
def data_process(path):
    # 该函数作用为读取txt，返回类似pandas的dataframe结构
    feature = np.genfromtxt(path, dtype=np.float32)
    # 标签个数
    num_feature = len(feature[0])
    # 转为pandas结构
    feature = pd.DataFrame(feature)
    # 数据切分
    label = feature.iloc[:, num_feature - 1]
    feature = feature.iloc[:, 0: num_feature - 2]
    return feature, label


if __name__ == '__main__':
    train_path = r"./data/train_feat.txt"
    test_path = r"./data/test_feat.txt"
    train_feature, train_label = data_process(train_path)
    test_feature, test_label = data_process(test_path)
    # 生成模型，注意里面的参数
    gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1
                                     , min_samples_split=2, min_samples_leaf=1, max_depth=3
                                     , init=None, random_state=None, max_features=None
                                     , alpha=0.9, verbose=0, max_leaf_nodes=None
                                     , warm_start=False)

    gbdt.fit(train_feature, train_label)
    pred = gbdt.predict(test_feature)
    total_error = 0
    for i in range(pred.shape[0]):
        print('pred:', pred[i], ' label:', test_label[i])
    print(mean_squared_error(test_label, pred))


