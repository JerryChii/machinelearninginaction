import os
import csv
import sys
from os.path import dirname, join

import numpy as np

def load_lr_data():
    module_path = dirname(__file__)
    base_dir = join(module_path, 'data')
    data = np.loadtxt(join(base_dir, 'lr_data.txt'))

    return data[:,:-1].copy(), data[:,-1].copy()

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def graDecent(dataMat,labelMat):
    m,n = dataMat.shape
    X0 = np.ones((m,1)).astype(dataMat.dtype)
    dataMat = np.mat(np.column_stack((X0, dataMat)))
    labelMat = np.mat(labelMat).T

    theta = np.mat(np.ones((dataMat.shape[1],1)))
    lambda_val = 0.1
    alpha = 0.001
    maxCycles = 500

    for k in range(maxCycles):
        h = sigmoid(dataMat * theta)
        err = (h - labelMat)
        theta = theta*(1-alpha*lambda_val/m) - alpha * dataMat.T * err

    return theta


dataMat, labelMat = load_lr_data()

print graDecent(dataMat, labelMat)

# print sigmoid(np.array([0,0]))