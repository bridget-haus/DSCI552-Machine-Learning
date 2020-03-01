#GOAL: Run the kmeans algorithm on the dataset using scikit-learn
import re
import numpy as np
import pandas as pd
from random import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#DATA CLEANING
def clean_data(txt_file):
    '''Clean out all the whitespace and newlines 
    Turn each datapoint into a list of two coordinates
    Add all datapoints to a list
    '''
    file = open(txt_file, "r")
    lines = file.readlines()
    list_rows = []
    for line in lines:
        item2 = line.strip()
        item3 = re.split(',',item2)
        item4 = []
        for num_str in item3:
            num_float = float(num_str)
            item4.append(num_float)
        list_rows.append(item4) #Append cleaned rows to new list
    return list_rows

def init_dataframe(list_rows, var1, var2):
    '''Converts list of rows into a dataframe. Assumes two variables. '''
    var1_list = []
    var2_list = [] #Assumes two data attributes only
    for l in list_rows:
        var1_list.append(l[0])
        var2_list.append(l[1])
    df = pd.DataFrame({var1: var1_list,
                       var2: var2_list})
    return df

def init_centroids(df, k, var1, var2):
    ''' Assigns centroids to a random location, based on variable min and max.
    Assumes two variables
    df - the training data in dataframe form
    k - the number of clusters
    '''
    var1_min = df[var1].min()
    var1_max = df[var1].max()
    var2_min = df[var2].min()
    var2_max = df[var2].max()

    centroid_list = []
    for cluster in range(k):
        centroid = []
        rand_val = random()
        centroid.append(var1_min + (rand_val * (var1_max - var1_min)))
        rand_val = random()
        centroid.append(var2_min + (rand_val * (var2_max - var2_min)))
        centroid_list.append(centroid)
    centroid_list_a = np.array(centroid_list)
    return centroid_list_a

#INITIALIZATION
txt_file = '../gmm/clusters.txt'
list_rows = clean_data(txt_file)

k = 3 #Number of specified clusters
var1 = 'var1'
var2 = 'var2'
colmap = {0: 'r', 1: 'g', 2: 'b'}
df = init_dataframe(list_rows, var1, var2)
centroids = init_centroids(df, k, var1, var2)

#FITTING THE KMEANS CLUSTERS
kmeans = KMeans(n_clusters=k, init=centroids, n_init=1, random_state=None)
kmeans.fit(df)

labels = kmeans.labels_.tolist() #Pull out the final data_labels
df['closest'] = labels #Add labels to dataframe, and label color
df['color'] = df['closest'].map(lambda x: colmap[x])

final_centroids = kmeans.cluster_centers_ #Pull out final centroid locations
print('Final cluster centroids:', final_centroids)

k_index_list = range(k)
final_centroids_d = {}
for k_index, centroid in zip(k_index_list, final_centroids):
    final_centroids_d[k_index] = list(centroid)

#VISUALIZING THE RESULT
fig = plt.figure(figsize=(5, 5))

#Plot datapoint and it's color, based on centroid
plt.scatter(df[var1], df[var2], color=df['color'], alpha=0.3, edgecolor='k')
plt.xlim(-5, 10)
plt.ylim(-5, 10)

#Plot centroids and their color
for i in final_centroids_d.keys():
    plt.scatter(*final_centroids_d[i], color=colmap[i])
plt.savefig('kmeans_scikit.png')