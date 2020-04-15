from sklearn import svm
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def main():

	data, labels = get_data('nonlinsep.txt')
	X_std, labels, svc = fit_svm(data, labels)
	plot(X_std, labels, svc)


def get_data(file):

	with open(file, "r") as fd:
		rows = fd.read().splitlines()
		data = []
		labels = []
		for row in rows:
			row_list = row.split(',')
			data.append(row_list[:-1])
			labels.append(row_list[-1])

	return data, labels


def fit_svm(data, labels):

	# Standarize features
	scaler = StandardScaler()
	X_std = scaler.fit_transform(data)


	# Create support vector classifier
	svc = svm.SVC(kernel='rbf', gamma=.1, C=10**(2))

	# Train model
	svc.fit(X_std, labels)
	print(f'there are {len(svc.support_vectors_)} support vectors: \n {svc.support_vectors_}')

	return X_std, labels, svc


def plot(X_std, labels, svc):

	# Plot data points and color using their class
	color = ['red' if c == '+1' else 'blue' for c in labels]
	plt.scatter(X_std[:,0], X_std[:,1], c=color)
	plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], marker='D', c='black')

	# Plot the hyperplane
	plt.axis("off"), plt.show()


if __name__ == '__main__':
	main()
