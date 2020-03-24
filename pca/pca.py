# PART 1: Do PCA on the 3d dataset and report the direction of each
# Report the direction of the first 2 Principal Components
# Code PCA algorithm from scratch

import os
import numpy as np
from sklearn.preprocessing import StandardScaler  # Used for standardizing the data
from numpy import linalg as la
import matplotlib.pyplot as plt


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

	# Standardize the dataset
	# This will shift data for each variable so that each variable has mean = 0, and st_dev = 1
	std = StandardScaler()
	data_list_std = std.fit_transform(data_list_np)

	return data_list_std


def pca_transform(data_list_std, k):
	''' Take in the dataset and reduce it to k dimensions using PCA method.
    data_list_std - standardized np array, output from clean_data()
    '''

	# Compute covariance matrix
	cov_data = np.cov(data_list_std, rowvar=False)

	# Compute e-values and e-vectors from covariance matrix
	e_values, e_vectors_t = la.eig(cov_data)

	e_vectors = np.transpose(e_vectors_t)
	first_vector = e_vectors[0].dot(-1)  # Fix the negative signs for first e-vector
	e_vectors2 = np.vstack((first_vector, e_vectors[1:]))
	e_vectors2_t = np.transpose(e_vectors2)

	print('PC Eigenvalues: ', e_values)
	print('PC Eigenvectors: ' + '\n', e_vectors2)

	# Construct a projection matrix W from the “top” k eigenvectors
	e_pairs = [(np.abs(e_values[i]), e_vectors2_t[:, i]) for i in range(len(e_values))]

	# Sort the (eigenvalue, eigenvector) pairs from high to low
	e_pairs.sort(key=lambda k: k[0], reverse=True)

	# Select the top two e_vector sets (k_dim) from e_pairs
	# w = np.hstack((e_pairs[0][1][:, np.newaxis], e_pairs[1][1][:, np.newaxis]))

	best_pairs = []
	for i in range(k):
		pair = e_pairs[i][1].reshape(len(cov_data), 1)
		best_pairs.append(pair)

	w = np.hstack(best_pairs)

	# Transform the d-dimensional input dataset, using the projection matrix W
	# to obtain the new k-dimensional feature subspace.
	data_list_pca = data_list_std.dot(w)
	print("Initial shape: ", data_list_std.shape)
	print('New data shape: ', data_list_pca.shape)

	# Output the data to a new txt file
	fd = open("pca_output.txt", "w")
	for row in data_list_pca:
		row1 = str(row)
		row2 = row1.replace("[", "")
		row3 = row2.replace("]", "")
		fd.write(str(row3) + "\n")
	fd.close()

	# Plot the transformed shape
	plt.scatter(data_list_pca[:, 0], data_list_pca[:, 1])
	plt.axis('equal');
	plt.show()


def main():
	data_text = 'pca-data.txt'
	data_list_np = clean_data(data_text)
	pca_transform(data_list_np, 2)


if __name__ == "__main__":
	main()