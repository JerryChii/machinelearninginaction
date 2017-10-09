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


def graDecent(dataArr,labelArr):
    m,n = dataArr.shape
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T

    theta = np.mat(np.ones((dataMat.shape[1],1)))
    lambda_val = 0.1
    alpha = 0.001
    maxCycles = 500

    for k in range(maxCycles):
        h = sigmoid(dataMat * theta)
        err = (h - labelMat)
        theta = theta*(1-alpha*lambda_val/m) - alpha * dataMat.T * err

    return theta


def stocGradAscent0():
    return 0



def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataArr,labelArr=load_lr_data()
    m = dataArr.shape[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(m):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)

    y = (-weights[0,0]-weights[1,0]*x)/weights[2,0]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

dataMat, labelMat = load_lr_data()

weights = graDecent(dataMat, labelMat)

plotBestFit(weights)