import scikit_tree
from sklearn.svm import SVC

class_names, feature_names, data_train, data_test, labels_train, labels_test = scikit_tree.split_train_test('data.csv')

def svm():
	clf = SVC(kernel = 'linear', random_state = 1)
	clf.fit(data_train, labels_train)
	scikit_tree.predict(clf, data_test, labels_test)

if __name__ == '__main__':
	svm()