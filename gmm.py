import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import math
import os
from kmeans import *
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


def main():

    script_dir = os.path.dirname(__file__)
    filename = 'clusters.txt'
    path = os.path.join(script_dir, filename)

    data_array = get_data(path)
    r_ic, pi, mu, cov = gmm(data_array, 3)
    print(f'means:\n{mu}\n\ncovariance matrices:\n{cov}\n\namplitudes:\n{pi}')
    plot_gmm(r_ic, data_array)


def expectation(data_array, r_ic, pi, mu, cov):

    num_rows = r_ic.shape[0]
    num_clusters = r_ic.shape[1]

    sum = 0
    for row in range(num_rows):
        for cluster in range(num_clusters):
            sum += r_ic[row, cluster] * math.log(pi[cluster])
            sum += r_ic[row, cluster] * multivariate_normal.logpdf(data_array[row], mean=mu[:, cluster], cov=cov[cluster])
    return sum


def gmm(data_array, num_clusters):

    # find shape of array
    num_rows = data_array.shape[0]
    num_columns = data_array.shape[1]


    cluster_list, cluster_assignment, centers = k_means(data_array, num_clusters)
    r_ic = np.zeros((num_rows, num_clusters))

    for i in range(len(r_ic)):
        r_ic[i][cluster_assignment[i] - 1] = 1

    init = False
    p = 0
    while True:
        # Update optimal {pi, mu, cov} given fixed {r_ic}
        sum_n_qnk = np.sum(r_ic, axis=0)
        pi = sum_n_qnk / num_rows
        mu = np.dot(data_array.T, r_ic) / sum_n_qnk
        cov = np.zeros((num_clusters, num_columns, num_columns))
        for cluster in range(num_clusters):
            for row in range(num_rows):
                XMu = np.reshape(data_array[row] - mu[:, cluster], (num_columns, 1))
                cov[cluster] += r_ic[row, cluster] * np.dot(XMu, XMu.T)
            cov[cluster] /= sum_n_qnk[cluster]

        # Check whether we have achieved stopping criterion
        if not init:
            LL = expectation(data_array, r_ic, pi, mu, cov)
            init = True
        else:
            oldLL = LL
            LL = expectation(data_array, r_ic, pi, mu, cov)
            if math.fabs(LL - oldLL) < 1e-1:
                break

        # Update optimal {r_ic} given fixed {pi, mu, cov}
        for row in range(num_rows):
            for cluster in range(num_clusters):
                Nprob = multivariate_normal.pdf(data_array[row], mean=mu[:, cluster], cov=cov[cluster])
                r_ic[row, cluster] = pi[cluster] * Nprob
            r_ic[row] /= np.sum(r_ic[row])

    return r_ic, pi, mu, cov


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


if __name__ == "__main__":
    main()