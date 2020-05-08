import scikit_tree
from sklearn.svm import SVC
import time

class_names, feature_names, data_train, data_test, labels_train, labels_test = scikit_tree.split_train_test('data.csv')

def svm():
	clf = SVC(kernel = 'linear', random_state = 1)

	#Train
	start = time.time()
	clf.fit(data_train, labels_train)
	end = time.time()
	train_time = round(((end - start) * 1000), 4)
	print(f'Training time: {train_time} ms')

	#Test
	# start = time.time()
	scikit_tree.predict(clf, data_test, labels_test)
	# end = time.time()
	# test_time = round(((end - start) * 1000), 4)
	# print(f'Testing time: {test_time} ms')

if __name__ == '__main__':
	svm()