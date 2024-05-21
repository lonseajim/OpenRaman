#
'''
    自适应迭代加权惩罚最小二乘法
'''

#!/usr/bin/python

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
import matplotlib
from matplotlib import pyplot as plt

def WhittakerSmooth(x, w, lambda_, differences=1):
    X = np.matrix(x)
    m = X.size
    i = np.arange(0, m)
    E = eye(m, format='csc')
    D = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * D.T * D))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)
def airPLS(x, lambda_=100, porder=1, itermax=15):
    m = x.shape[0]#计算长宽维度
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax):
                print('WARING max iteration reached!')
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    z=d
    return z

# data = pd.read_csv(r"D:\python-c\拉曼光谱预处理（批处理）算法总结\H_19_去背景算法\E1-01.txt", sep='\t', header=None)
# data = np.array(data)
# x = data[:, 0]
# y = data[:, 1]
# Y = airPLS(y, lambda_=100, porder=1, itermax=15)
#
# plt.plot(x,Y, linewidth=1)
# plt.show()
