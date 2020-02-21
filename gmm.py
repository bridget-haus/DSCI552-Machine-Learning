import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import os
from kmeans import *
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import random
import operator
import copy
matplotlib_axes_logger.setLevel('ERROR')


def main():

    script_dir = os.path.dirname(__file__)
    filename = 'clusters.txt'
    path = os.path.join(script_dir, filename)

    data_array = get_data(path)
    ric, pi, mu, cov, num_runs = gmm(data_array, 3, 0.0005)
    print(f'means:\n{mu}\n\ncovariance matrices:\n{cov}\n\namplitudes:\n{pi}')
    print('Runs: ', num_runs)
    plot_gmm(ric, data_array, block=True)



def gmm(data_array, num_clusters, converge_thresh):


    num_rows = data_array.shape[0]
    num_columns = data_array.shape[1]

    ric = []

    for i in range(0,num_rows) :
        ric.append([])
        size = 100
        for j in range(0, num_clusters - 1) :
            rand_num= random.randint(0, size)
            size -= rand_num
            ric[i].append(float(rand_num/100))
        ric[i].append(float(size / 100))

    num_iterations = 0
    max_dif = 1

    while abs(max_dif) > converge_thresh:

        sum_clusters = np.sum(ric,0)

        pi = sum_clusters / num_rows

        mu = np.dot(data_array.T, np.array(ric)) / sum_clusters

        cov = np.zeros((num_clusters, num_columns, num_columns))
        for cluster in range(num_clusters):
            for row in range(num_rows):

                data_Mu = np.reshape(data_array[row] - mu[:, cluster], (num_columns, 1))
                cov[cluster] += ric[row][cluster] * np.dot(data_Mu, data_Mu.T)

            cov[cluster] /= sum_clusters[cluster]

        for row in range(num_rows):
            for cluster in range(num_clusters):

                Nprob = multivariate_normal.pdf(data_array[row], mean=mu[:, cluster], cov=cov[cluster])

                ric[row][cluster] = pi[cluster] * Nprob

            ric[row] /= np.sum(ric[row])

        #check to see if converged to some threshold
        if num_iterations != 0 :
            diff = list(map(operator.sub, ric, old_cluster_list))
            max_dif = max(map(lambda x: x[num_clusters - 1], diff), key=abs)


        #show the progess of the clustering with noticable differences
        if num_iterations % 10 == 0 and abs(max_dif) > converge_thresh * 10:
            plot_gmm(ric, data_array)


        num_iterations += 1
        old_cluster_list = copy.deepcopy(ric)


    return ric, pi, mu, cov, num_iterations


def plot_gmm(ric, data_array, block=False):

    counter = 0
    for point in data_array:
        x = point[0]
        y = point[1]
        r = 1 * ric[counter][0]
        g = 1 * ric[counter][1]
        b = 1 * ric[counter][2]
        plt.scatter([x], [y], c=(r, g, b))
        counter = counter + 1

    if not(block) :
        plt.show(block=False)
        plt.pause(.1)
    else :
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()