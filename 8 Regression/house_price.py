# -*- coding: utf-8 -*-
"""
@Time    : 2020-04-11 21:21
@Author  : Veigar
@FileName: house_price.py
@Github  ：https://github.com/veigaran
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

# 读取数据
# 训练集
house = pd.read_csv("data\kc_train.csv")
target = pd.read_csv('data\kc_train2.csv')
# 测试集
t = pd.read_csv('data\kc_test.csv')

# 查看信息
house.info()
from sklearn.preprocessing import MinMaxScaler

# 特征缩放
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(house)
# 内部拟合
scaler_house = minmax_scaler.transform(house)
scaler_house = pd.DataFrame(scaler_house, columns=house.columns)

# 测试集
mm = MinMaxScaler()
mm.fit(t)
scaler_t = mm.transform(t)
scaler_t = pd.DataFrame(scaler_t, columns=t.columns)

from sklearn.linear_model import LinearRegression

# 选择基于梯度下降的线性回归模型
lr_reg = LinearRegression()
# 进行拟合
lr_reg.fit(scaler_house, target)

# 使用均方误差用于评价模型好坏
from sklearn.metrics import mean_squared_error

# 输入数据进行预测得到结果
preds = lr_reg.predict(scaler_house)
# 使用均方误差来评价模型好坏，可以输出mse进行查看评价值
mse = mean_squared_error(preds, target)

# 作图
plot.figure(figsize=(10, 7))
num = 100
x = np.arange(1, num + 1)
plot.plot(x, target[:num], label='target')
plot.plot(x, preds[:num], label='preds')
plot.legend(loc='upper right')
plot.show()

result = lr_reg.predict(scaler_t)
df_result = pd.DataFrame(result)
df_result.to_csv("data/result.csv")
