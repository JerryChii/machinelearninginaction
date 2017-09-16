#coding=utf-8
import os
import csv
import sys
import random
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
        # theta = theta*(1-alpha*lambda_val/m) - tmp
        # theta0不用正则化，一下写的太丑了，需要改
        tmp = alpha * dataMat.T * err
        theta[0,0] = theta[0,0] - tmp[0,0]
        theta[1:,:] = theta[1:,:]*(1-alpha*lambda_val/m) - tmp[1:,:]

    return theta


# 1、初始化参数theta为1
# 2、循环每行，根据公式更新每个参数
# 3、调alpha参数，之前用0.001测试偏的太多
def stocGraDecent0(dataNdArr,labelMatNdArr):
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

# 1、在随机梯度下降的基础上迭代numIter次
# 2、每次更新参数的时候，随机从m个样本里选取一条数据更新(每条样本只能使用一次)
# 3、每次更新参数的时候，修改alpha的值(减少)，比采用固定alpha的方法收敛速度更快
def stocGraDecent1(dataNdArr,labelMatNdArr, numIter=150):
    m, n = dataNdArr.shape
    theta = np.ones(n)
    # lambda_val = 0.001

    for i in range(numIter):
        dataIndex = range(m)
        for j in range(m):
            # 虽然alpha会随着迭代次数不断减小，但永远不会减小到0
            alpha = 4.0/(i+j+1.0) + 0.0001
            # 通过随机选取样本来更新回归系数将减少周期性的波动
            randIndex = random.randrange(0, len(dataIndex))
            h = sigmoid(sum(dataNdArr[randIndex]*theta))
            err = h - labelMatNdArr[randIndex]
            # theta = theta*(1-alpha*lambda_val) - alpha*err*dataNdArr[randIndex]
            theta = theta - alpha*err*dataNdArr[randIndex]
            del(dataIndex[randIndex])

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
weights = graDecent(dataNdArr,labelMatNdArr)
plotBestFit(np.reshape(weights.tolist(), (3,)))

#weights = stocGraDecent0(dataNdArr,labelMatNdArr)
#plotBestFit(weights)

#weights = stocGraDecent1(dataNdArr,labelMatNdArr)
#plotBestFit(weights)