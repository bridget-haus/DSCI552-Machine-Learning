import numpy as np
import matplotlib.pyplot as plt
import random
import os


def main():

    script_dir = os.path.dirname(__file__)
    filename = 'clusters.txt'
    path = os.path.join(script_dir, filename)

    data_array = get_data(path)
    cluster_list, cluster_assignment = k_means(data_array, 3)
    plot_kmeans(cluster_list)


def get_data(path):

    with open(path) as f:
        lines = f.readlines()
        # assign each x coordinate to a list
        x = [float(line.split(',')[0]) for line in lines]
        # assign each y coordinate to a list
        y = [float(line.split(',')[1]) for line in lines]

    # create numpy array for our dataset
    data = list(zip(x, y))
    data_array = np.asarray(data, dtype=np.float32)

    return data_array


def random_cluster_center(data_array, num_clusters):

    # find shape of array
    num_rows = data_array.shape[0]
    num_columns = data_array.shape[1]

    centers = np.zeros((num_clusters, num_columns))

    # generate random cluster centers
    dup_check = []
    for i in range(num_clusters):

        randint = random.randint(0, num_rows - 1)
        while randint in dup_check:
            randint = random.randint(0, num_rows - 1)
        dup_check.append(randint)
        # append random centers to centers matrix
        centers[i] = data_array[randint]

    return num_rows, num_columns, centers


def k_means(data_array, num_clusters):

    num_rows, num_columns, centers = random_cluster_center(data_array, 3)

    cluster_assignment = np.zeros(num_rows, dtype=int)

    while True:

        j = 0
        changed = False
        cluster_list = []

        for i in range(num_clusters):
            cluster_list.append([])

        for point in data_array:
            # argmin is the label of the cluster
            argmin = 1
            dist_list = []
            for mean in centers:
                # calculate distance between each point and each cluster mean
                dist = np.linalg.norm(point - mean)
                dist_list.append([dist, argmin])
                argmin = argmin + 1

            # for each point, choose cluster who's distance is smallest
            min_dist = min(dist_list)
            argmin = min_dist[1]

            # reassign point to the closest cluster mean
            if cluster_assignment[j] != argmin:
                changed = True
                cluster_assignment[j] = argmin
            cluster_list[argmin - 1].append(point)
            j += 1

        # reached convergence when no point is reassigned
        if not changed:
            break

        counter = 0
        for cluster in cluster_list:

            # recalculate cluster centers
            if len(cluster) != 0:
                sum_matrix = sum(cluster)
                sum_matrix /= len(cluster)
                centers[counter] = sum_matrix
                counter = counter + 1

    return cluster_list, cluster_assignment


def plot_kmeans(cluster_list):

    colors = ['red', 'blue', 'green']
    counter = 0

    # plot all of the clusters
    for cluster in cluster_list:
        x = [float(point[0]) for point in cluster]
        y = [float(point[1]) for point in cluster]
        plt.scatter(x, y, color=colors[counter])

        counter = counter + 1
    plt.show()


if __name__ == "__main__":
    main()