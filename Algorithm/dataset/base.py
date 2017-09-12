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



## test
data, label = load_lr_data()
print data.shape
print label.shape