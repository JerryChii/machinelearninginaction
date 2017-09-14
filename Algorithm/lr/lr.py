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


# print sigmoid(np.array([0,0]))