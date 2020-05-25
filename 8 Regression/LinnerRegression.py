# -*- coding: utf-8 -*-
"""
@Time    : 2020-04-11 16:40
@Author  : Veigar
@FileName: LinnerRegression.py
@Github  ：https://github.com/veigaran
"""
from numpy import *
import matplotlib.pylab as plt

# 载入数据
def load_data(file_name):
    data_mat = []
    label_mat = []
    num_feature = len(open(file_name).readline().split('\t')) - 1
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feature):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def stand_regression(x, y):
    x_mat = mat(x)
    y_mat = mat(y).T
    xTx = x_mat.T * x_mat
    if linalg.det(xTx) == 0.0:
        print("Cannot do inverse")
        return
    ws = xTx.I * (x_mat.T * y_mat)
    return ws


def regression_1():
    x, y = load_data("data/data.txt")
    ws = stand_regression(x, y)
    x_mat = mat(x)
    y_mat = mat(y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_mat[:, 1].flatten(), y_mat.T[:, 0].flatten().A[0])
    x_copy = x_mat.copy()
    x_copy.sort(0)
    y_hat = x_copy * ws
    ax.plot(x_copy[:, 1], y_hat)
    plt.show()


def lwlr(test_point, x, y, k=1.0):
    x_mat = mat(x)
    y_mat = mat(y).T
    m = shape(x_mat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diff_mat = test_point - x_mat[j, :]
        weights[j, j] = exp(diff_mat * diff_mat.T / (-2.0 * k ** 2))
    xTx = x_mat.T * (weights * y_mat)
    if linalg.det(xTx) == 0.0:
        print("Cannot do inverse")
        return
    ws = xTx.I * (x_mat.T * (weights * y_mat))
    return test_point * ws


def lwlr_test(test, x, y, k=1.0):
    m = shape(test)[0]
    y_hat = zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(test[i], x, y, k)
    return y_hat


def regression_2():
    x, y = load_data("data/data.txt")
    y_hat = lwlr_test(x, x, y, 0.01)
    x_mat = mat(x)
    srt_index = x_mat[:, 1].argsort(0)
    x_sort = x_mat[srt_index][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_sort[:, 1], y_hat[srt_index])
    ax.scatter(x_mat[:, 1].flatten().A[0], mat(y).T.flatten().A[0], s=2, c='red')
    plt.show()


def ridge_regression(x, y, lam=0.2):
    xTx = x.T * x
    denom = xTx + eye(shape(x)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (x.T * y)
    return ws


def ridge_test(x, y):
    x_mat = mat(x)
    y_mat = mat(y).T
    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_means = mean(x, 0)
    x_var = var(x_mat, 0)
    x_mat = (x_mat - x_means) / x_var
    num_test_points = 30
    w_mat = zeros((num_test_points, shape(x_mat)[1]))
    for i in range(num_test_points):
        ws = ridge_regression(x_mat, y_mat, exp(i - 10))
        w_mat[i, :] = ws.T
    return w_mat


def regression_3():
    abX, abY = load_data("data/8.Regression/abalone.txt")
    ridgeWeights = ridge_test(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()


if __name__ == "__main__":
    regression_1()
    # regression2()
    # abaloneTest()
    # regression3()