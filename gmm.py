import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import math
import os
from kmeans import *
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


# Calculate negative log likelihood

def calc_LL(data_array, r_ic, Pi, Mu, Sigma):

    num_rows = r_ic.shape[0]
    num_clusters = r_ic.shape[1]

    sum = 0
    for n in range(num_rows):
        for k in range(num_clusters):
            sum += r_ic[n, k] * math.log(Pi[k])
            sum += r_ic[n, k] * multivariate_normal.logpdf(data_array[n], mean=Mu[:, k], cov=Sigma[k])
    return sum


def gmm(data_array, num_clusters):

    # find shape of array
    num_rows = data_array.shape[0]
    num_columns = data_array.shape[1]

    # Initialize {r_ic}, {Pi, Mu, Sigma}.
    # Here we run KMeans to cluster and use the result as an initialization of {r_ic},
    # and {Pi, Mu, Sigma} are implicit from the resulting {r_ic}.
    cluster_list, cluster_assignment = k_means(data_array, num_clusters)
    r_ic = np.zeros((num_rows, num_clusters))

    for i in range(len(r_ic)):
        r_ic[i][cluster_assignment[i] - 1] = 1

    # Perform coordinate ascent until difference
    # in log-likelihood is small (here: < 1e-1)
    init = False
    p = 0
    while True:
        # Update optimal {Pi, Mu, Sigma} given fixed {r_ic}
        sum_n_qnk = np.sum(r_ic, axis=0)
        Pi = sum_n_qnk / num_rows
        Mu = np.dot(data_array.T, r_ic) / sum_n_qnk
        Sigma = np.zeros((num_clusters, num_columns, num_columns))
        for k in range(num_clusters):
            for n in range(num_rows):
                XMu = np.reshape(data_array[n] - Mu[:, k], (num_columns, 1))
                Sigma[k] += r_ic[n, k] * np.dot(XMu, XMu.T)
            Sigma[k] /= sum_n_qnk[k]

        # Check whether we have achieved stopping criterion
        if not init:
            LL = calc_LL(data_array, r_ic, Pi, Mu, Sigma)
            init = True
        else:
            oldLL = LL
            LL = calc_LL(data_array, r_ic, Pi, Mu, Sigma)
            if math.fabs(LL - oldLL) < 1e-1:
                break

        # Update optimal {r_ic} given fixed {Pi, Mu, Sigma}
        for n in range(num_rows):
            for k in range(num_clusters):
                Nprob = multivariate_normal.pdf(data_array[n], mean=Mu[:, k], cov=Sigma[k])
                r_ic[n, k] = Pi[k] * Nprob
            r_ic[n] /= np.sum(r_ic[n])

    return r_ic


def plot_gmm(r_ic, data_array):

    counter = 0
    for point in data_array:
        x = point[0]
        y = point[1]
        r = 1 * r_ic[counter][0]
        g = 1 * r_ic[counter][1]
        b = 1 * r_ic[counter][2]
        plt.scatter([x], [y], c=(r, g, b))
        counter = counter + 1
    plt.show()


script_dir = os.path.dirname(__file__)
filename = 'clusters.txt'
path = os.path.join(script_dir, filename)

data_array = get_data(path)

r_ic = gmm(data_array, 3)
plot_gmm(r_ic, data_array)
