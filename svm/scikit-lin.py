from sklearn import svm
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def main():

	data, labels = get_data('linsep.txt')
	X_std, labels, xx, yy, svc, w, b = fit_svm(data, labels)
	equation_of_line(w, b)
	plot(X_std, labels, xx, yy, svc)


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
	svc = svm.SVC(kernel='linear', C=1000)

	# Train model
	svc.fit(data, labels)
	print(f'there are {len(svc.support_vectors_)} support vectors: \n{svc.support_vectors_}')

	# Create the hyperplane
	w = svc.coef_[0]
	print(f'the weights are: \n{w}')
	a = -w[0] / w[1]
	xx = np.linspace(-0.05, 1)
	yy = a * xx - (svc.intercept_[0]) / w[1]

	b = svc.intercept_
	print(f'The intercept is: \n{b}')

	return data, labels, xx, yy, svc, w, b


def plot(data, labels, xx, yy, svc):

	# Plot data points and color using their class
	color = ['red' if c == '+1' else 'blue' for c in labels]
	for point in data:
		x = [float(point[0]) for point in data]
		y = [float(point[1]) for point in data]
		plt.scatter(x, y, color = color)

	plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], marker='D', c='black')

	# Plot the hyperplane
	plt.plot(xx, yy)
	plt.axis("off"), plt.show()


def equation_of_line(weights, intercept):

	m = -weights[0]/weights[1]
	b = -intercept / weights[1]

	print(f'y = {m}x + {b[0]}')


if __name__ == '__main__':
	main()
