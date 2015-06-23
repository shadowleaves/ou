# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 12:01:08 2015

@author: han.yan
"""

# from numba import njit
import pandas as pd
import numpy as np
# from sklearn.decomposition import PCA
from scipy.linalg import svd, eigh
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.vector_ar.var_model import VAR

# from utils.matrix import pca_cov

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', '{:,g}'.format)
np.set_printoptions(precision=4, suppress=True)
np.random.seed()


def pca_cov(cov_mat, k=None):
    # Compute the covariance matrix
    # cov_mat = np.cov(X.T)
    if k is None:
        k = cov_mat.shape[0]
    # Eigendecomposition of the covariance matrix
    eigval, eigvec = eigh(cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    # and sort the (eigenvalue, eigenvector) tuples from high to low
    ind = np.argsort(eigval)[::-1][:k]

    w = -eigvec.T[ind]   # loadings, i.e. the right eigen vector v in svd
    e = eigval[ind]

    return w, e


def fit_ou(mat, p):
    mat = resid.values * 1e4
    model = VAR(mat)
    res = model.fit(maxlags=p)
    res.plot()


if __name__ == '__main__':

    k = 4
    x = pd.read_csv('d:/.home/cache/tmp/rates.csv', header=None)
    x = np.log(x)  # .diff()[1:]
    # x = x - x.mean()
    cov_mat = np.cov(x.T)
    w, e = pca_cov(cov_mat, k)
    xp = np.dot(x, w.T)  # projected
    resid = x - np.dot(xp, w)

    print xp

    plt.plot(resid)

    for i in xrange(k):
        y = xp[:, i] * 1e4
        # y = resid[i].values
        # plt.plot(y)
        model = AR(y)
        res = model.fit(maxlag=1)
        c, a = res.params
        tau = 1. / 252
        theta = -np.log(a) / tau
        mu = c / (1 - a)

        cov = np.cov(res.resid)
        tst = theta * 2
        sigma2 = tst * cov / (1 - np.exp(-tst * tau))
        # sigma = np.sqrt(sigma2)

        y_sd = np.sqrt(sigma2 / (2 * theta))
        print '%.1f %.4f %.2f' % (mu, theta, y_sd)
        plt.figure(figsize=(5, 8))
        plt.subplot(k * 100 + 10 + i + 1)
        plt.plot((y - mu) / y_sd)

    if False:
        u, d, v = svd(x - x.mean(), full_matrices=False)
        u = u[:, :k]
        d = d[:k]
        v = v[:k]

        score = np.dot(u, np.diag(d))
        loadings = v

        print score
        print loadings
