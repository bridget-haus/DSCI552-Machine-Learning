import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import math
import os
from kmeans import *
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

#Calculate negative log likelihood
def calc_LL(X, Q, Pi, Mu, Sigma):
    N = Q.shape[0]
    K = Q.shape[1]
    sum = 0
    for n in range(N):
        for k in range(K):
            sum += Q[n, k] * math.log(Pi[k])
            sum += Q[n, k] * multivariate_normal.logpdf(X[n], mean=Mu[:, k], cov=Sigma[k])
    return sum


def gmm(X, K):

    # init params
    N = X.shape[0]
    D = X.shape[1]

    # Initialize {Q}, {Pi, Mu, Sigma}.
    # Here we run KMeans to cluster and use the result as an initialization of {Q},
    # and {Pi, Mu, Sigma} are implicit from the resulting {Q}.
    _, Z = k_means(X, K)
    Q = np.zeros((N, K))
    counter = 0
    for i in range(len(Q)):
        Q[i][Z[i] - 1] = 1


    # Perform coordinate ascent until difference
    # in log-likelihood is small (here: < 1e-1)
    init = False
    p = 0
    while True:
        # Update optimal {Pi, Mu, Sigma} given fixed {Q}
        sum_n_qnk = np.sum(Q, axis=0)
        Pi = sum_n_qnk / N
        Mu = np.dot(X.T, Q) / sum_n_qnk
        Sigma = np.zeros((K, D, D))
        for k in range(K):
            for n in range(N):
                XMu = np.reshape(X[n] - Mu[:, k], (D, 1))
                Sigma[k] += Q[n, k] * np.dot(XMu, XMu.T)
            Sigma[k] /= sum_n_qnk[k]

        # Check whether we have achieved stopping criterion
        if not init:
            LL = calc_LL(X, Q, Pi, Mu, Sigma)
            init = True
        else:
            oldLL = LL
            LL = calc_LL(X, Q, Pi, Mu, Sigma)
            if math.fabs(LL - oldLL) < 1e-1:
                break

        # Update optimal {Q} given fixed {Pi, Mu, Sigma}
        for n in range(N):
            for k in range(K):
                Nprob = multivariate_normal.pdf(X[n], mean=Mu[:, k], cov=Sigma[k])
                Q[n, k] = Pi[k] * Nprob
            Q[n] /= np.sum(Q[n])

    return Q

def plot_gmm(Q, data_array):

    colors = ['red', 'blue', 'green']
    counter = 0

    for point in data_array:
        x = point[0]
        y = point[1]
        r = 1*Q[counter][0]
        g = 1*Q[counter][1]
        b = 1*Q[counter][2]
        plt.scatter([x], [y], c=(r, g, b))
        counter = counter + 1
    plt.show()


script_dir = os.path.dirname(__file__)
filename = 'clusters.txt'
path = os.path.join(script_dir, filename)

data_array = get_data(path)


Q = gmm(data_array, 3)
plot_gmm(Q, data_array)

# Softmax a row vector
# def __softmax(self, row):
#     sum = np.sum(np.exp(row))
#     return np.exp(row) / sum