#coding=utf-8
import os
import csv
import sys
from os.path import dirname, join

import numpy as np

def load_lr_data():
    module_path = dirname(__file__)
    base_dir = join(module_path, 'data')
    data = np.loadtxt(join(base_dir, 'lr_data.txt'))
    m = data.shape[0]
    X0 = np.ones((m, 1)).astype(data.dtype)
    return np.column_stack((X0, data[:,:-1])), data[:,-1].copy()

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def graDecent(dataNdArr,labelMatNdArr):
    m,n = dataNdArr.shape

    dataMat = np.mat(dataNdArr)
    labelMat = np.mat(labelMatNdArr).T

    theta = np.mat(np.ones((dataMat.shape[1],1)))
    lambda_val = 0.1
    alpha = 0.001
    maxCycles = 500

    for k in range(maxCycles):
        h = sigmoid(dataMat * theta)
        err = (h - labelMat)
        theta = theta*(1-alpha*lambda_val/m) - alpha * dataMat.T * err

    return theta


# 1、初始化参数theta为1
# 2、循环每行，根据公式更新每个参数
# 3、调alpha参数，之前用0.001测试偏的太多
def stocGraDecent(dataNdArr,labelMatNdArr):
    m, n = dataNdArr.shape
    theta = np.ones(n)
    alpha = 0.01
    #lambda_val = 0.001

    for i in range(m):
        h = sigmoid(sum(dataNdArr[i]*theta))
        err =  h - labelMatNdArr[i]
        #theta = theta*(1-alpha*lambda_val) - alpha*err*dataNdArr[i]
        theta = theta - alpha*err*dataNdArr[i]

    return theta


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataNdArr, labelMatNdArr=load_lr_data()

    n = dataNdArr.shape[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMatNdArr[i])== 1:
            xcord1.append(dataNdArr[i,1]); ycord1.append(dataNdArr[i,2])
        else:
            xcord2.append(dataNdArr[i,1]); ycord2.append(dataNdArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

dataNdArr,labelMatNdArr = load_lr_data()
# weights.shape: (3,1) 转成(3,)
#weights = graDecent(dataNdArr,labelMatNdArr)
#plotBestFit(np.reshape(weights.tolist(), (3,)))

weights = stocGraDecent(dataNdArr,labelMatNdArr)
plotBestFit(weights)
