import numpy as np
import matplotlib.pyplot as plt
import random

with open('/Users/bridgethaus/Desktop/clusters.txt') as f:
    lines = f.readlines()
    # assign each x coordinate to a list
    x = [float(line.split(',')[0]) for line in lines]
    # assign each y coordinate to a list
    y = [float(line.split(',')[1]) for line in lines]


# create numpy array for our dataset

data = list(zip(x,y))
data_array = np.asarray(data, dtype=np.float32)



def k_means(data_array, num_clusters):

    # find shape of array

    num_rows = data_array.shape[0]
    num_columns = data_array.shape[1]

    # set matrices to 0

    cluster_assignment = np.zeros(num_rows, dtype=int)
    means = np.zeros((num_clusters, num_columns))

    dup_check = []

    # generate random cluster centers
    for i in range(num_clusters):

        randint = random.randint(0, num_rows - 1)
        while randint in dup_check:
            randint = random.randint(0, num_rows - 1)
        dup_check.append(randint)
        means[i] = data_array[randint]

    while True:

        j = 0
        changed = False

        cluster_list = []

        for i in range(num_clusters):
            cluster_list.append([])

        for point in data_array:
            argmin = 1
            dist_list = []
            for mean in means:
                # calculate distance between each point and each cluster mean
                dist = np.linalg.norm(point - mean)
                dist_list.append([dist, argmin])
                argmin = argmin + 1

            min_dist = min(dist_list)
            argmin = min_dist[1]

            # reassign point to the closest cluster mean
            if cluster_assignment[j] != argmin:
                changed = True
                cluster_assignment[j] = argmin
            cluster_list[argmin - 1].append(point)
            j += 1

        # if no point has its cluster reassigned, we have converged
        if not changed:
            break

        counter = 0
        for cluster in cluster_list:

            # recalculate cluster means
            if len(cluster) != 0:
                sum_matrix = sum(cluster)
                sum_matrix /= len(cluster)
                means[counter] = sum_matrix
                counter = counter + 1

    return cluster_list, cluster_assignment


cluster_list, cluster_assignment = k_means(data_array, 3)


# plot all of the clusters

one = cluster_list[0]
two = cluster_list[1]
three = cluster_list[2]

x1 = [float(point[0]) for point in one]
y1 = [float(point[1]) for point in one]

x2 = [float(point[0]) for point in two]
y2 = [float(point[1]) for point in two]

x3 = [float(point[0]) for point in three]
y3 = [float(point[1]) for point in three]


plt.scatter(x1, y1, color='red')
plt.scatter(x2, y2, color='blue')
plt.scatter(x3, y3, color='green')

plt.show()