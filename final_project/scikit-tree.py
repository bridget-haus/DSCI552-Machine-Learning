import csv
from sklearn import tree
import graphviz
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def main():

	class_names, feature_names, data_train, data_test, labels_train, labels_test = split_train_test('data.csv')
	clf = decision_tree(class_names, feature_names, data_train, labels_train)
	predict(clf, data_test, labels_test)


def split_train_test(filename):

	with open(filename, 'r') as file:

		reader = csv.reader(file, delimiter=',')

		feature_names = next(reader, None)[2:-1]
		class_names = ['malignant', 'benign']

		data = []
		labels = []
		for row in reader:
			# create list of data, skip first column (id) and second column (label)
			data.append(list(row[2:]))
			# create list of labels, second column in csv
			labels.append(row[1])

	# randomly split train and test data
	data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.33)

	return class_names, feature_names, data_train, data_test, labels_train, labels_test


def decision_tree(class_names, feature_names, data_train, labels_train):

	# fit data
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(data_train, labels_train)
	# create tree visualization
	tree.export_graphviz(clf, out_file="tree.dot", filled=True, feature_names=feature_names, class_names=class_names)
	# convert .dot file to .png
	os.system('dot -Tpng tree.dot -o tree.png')
	return clf


def predict(clf, data_test, labels_test):

	# get predictions for test
	label_pred = clf.predict(data_test)

	print(f'Accuracy of model is: {accuracy_score(labels_test, label_pred)}')

	# generate confision matrix
	cm = confusion_matrix(labels_test, label_pred)

	# plot confusion matrix
	ax = plt.subplot()
	sns.heatmap(cm, annot=True, ax=ax, fmt="d");  # annot=True to annotate cells
	ax.set_xlabel('Predicted labels')
	ax.set_ylabel('True labels')
	ax.set_title('Confusion Matrix')
	ax.xaxis.set_ticklabels(['benign', 'malignant'])
	ax.yaxis.set_ticklabels(['benign', 'malignant'])
	plt.show()


if __name__ == '__main__':
	main()