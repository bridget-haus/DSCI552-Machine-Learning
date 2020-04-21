from sklearn import svm
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def main():
	data, labels = get_data('nonlinsep.txt')
	data, labels, svc = fit_svm(data, labels)
	plot(data, labels, svc)

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
	# Create support vector classifier
	svc = svm.SVC(kernel='rbf', gamma=.01, C=1000)

	# Train model
	svc.fit(data, labels)
	print(f'there are {len(svc.support_vectors_)} support vectors: \n {svc.support_vectors_}')
	
	return data, labels, svc

def plot(data, labels, svc):
	# Plot data points and color using their class
	color = ['red' if c == '+1' else 'blue' for c in labels]
	for point in data:
		x = [float(point[0]) for point in data]
		y = [float(point[1]) for point in data]
		plt.scatter(x, y, color=color)
	plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], marker='D', c='black')

	# Plot the hyperplane
	plt.axis("off"), plt.show()

if __name__ == '__main__':
	main()
