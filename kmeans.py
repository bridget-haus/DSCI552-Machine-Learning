import numpy as np
import matplotlib.pyplot as plt
import random

with open('/Users/bridgethaus/Desktop/clusters.txt') as f:
    lines = f.readlines()
    x = [float(line.split(',')[0]) for line in lines]
    y = [float(line.split(',')[1]) for line in lines]


#plt.scatter(x, y)
#plt.show()


X = list(zip(x,y))
X = np.asarray(X, dtype=np.float32)


def fit(X, K):

    N = X.shape[0]
    D = X.shape[1]

    z = np.zeros(N, dtype=int)
    means = np.zeros((K, D))

    S = []

    for i in range(K):

        randint = random.randint(0, N - 1)
        while randint in S:
            randint = random.randint(0, N - 1)
        S.append(randint)
        means[i] = X[randint]

    while True:
        j = 0
        changed = False
        cluster_list = []
        for i in range(K):
           cluster_list.append([])
        for datum in X:
            argmin = 1
            mindist = 0
            init = False
            for mean in means:
                dist = np.linalg.norm(datum - mean)
                if not init:
                    mindist = dist
                    init = True
                elif dist < mindist:
                    argmin = argmin + 1
                    mindist = dist

            if z[j] != argmin:
                changed = True
                z[j] = argmin
            cluster_list[argmin - 1].append(datum)
            j += 1

        if not changed:
            break

        counter = 0
        for cluster in cluster_list:
            if len(cluster) != 0:
                sum_matrix = sum(cluster)
                sum_matrix /= len(cluster)
                means[counter] = sum_matrix
                counter = counter + 1

    return cluster_list, z

cluster_list, z = fit(X, 3)

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