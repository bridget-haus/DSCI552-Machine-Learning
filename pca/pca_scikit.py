#PART 2: Do PCA on the 3d dataset and report the direction of each
#Report the direction of the first 2 Principal Components
#Use scikit to perform PCA algorithm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def clean_data(data_text):
    '''Returns a np array of datapoints and their attributes
    data_text - input data file, containing rows and columns of floating points
    '''
    with open(data_text) as f:
        lines = f.read().splitlines()
        data_list = []
        for line in lines:
            line = line.split('\t')
            line = [float(i) for i in line]
            data_list.append(line)
    
    data_list_np = np.array([np.array(xi) for xi in data_list])

    #Standardize data before finding cov matrix
    std = StandardScaler()
    data_list_std = std.fit_transform(data_list_np)

    return data_list_std

def fit_pca(n_components, data_list_std):
    ''' Calculate principal components of a numeric dataset.
    n_components - number of initial dimensions
    data_list_std - standardized np array, output from clean_data()
    '''
    pca = PCA(n_components=n_components)

    pca.fit(data_list_std)
    print('PC Eigenvalues: ', pca.explained_variance_) #PC Eigenvalues sorted descending
    print('PC Eigenvectors: ' + '\n', pca.components_) #PC Eigenvectors (direction), sorted by descending eigenvalues

def transform_pca(k_components, data_list_std):
    ''' Transform dataset to k dimensions using PCA
    k_components - the # of components to reduce the dataset to 
    data_list_std - standardized np array, output from clean_data()
    '''
    pca = PCA(n_components=k_components)
    pca.fit(data_list_std)
    data_trans = pca.transform(data_list_std)
    print("Initial shape: ", data_list_std.shape)
    print("Transformed shape: ", data_trans.shape)
    return data_trans

def main():
    data_text = 'pca-data.txt'
    data_list_std = clean_data(data_text)
    
    #Fit PCA on initial dataset
    n_components = len(data_list_std[0]) #Length of input data columns
    fit_pca(n_components, data_list_std)
    
    #Use PCA to transform dataset from 3D to 2D and output results
    k_components = 2
    data_trans = transform_pca(k_components, data_list_std)

    fd = open("pca_scikit_output.txt", "w")
    for row in data_trans:
        row1 = str(row)
        row2 = row1.replace("[","")
        row3 = row2.replace("]","")
        fd.write(str(row3) + "\n")
    fd.close()
    
    #Plot the transformed shape
    plt.scatter(data_trans[:, 0], data_trans[:, 1])
    plt.axis('equal');
    plt.show()

if __name__ == "__main__":
    main()