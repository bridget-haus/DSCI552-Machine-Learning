import time
import scikit_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class_names, feature_names, data_train, data_test, labels_train, labels_test = scikit_tree.split_train_test('data.csv')

#Random state tests
GLO = time.time()

#N_estimators tests
def n_est_tests(n_est_list):
	for n_est in n_est_list:
		start_time = time.time()
		model = RandomForestClassifier(criterion="entropy", n_estimators=n_est, random_state=0)
		model.fit(data_train, labels_train)
		end_time = time.time()
		t_time = round(((end_time - start_time) * 1000), 4)
		label_pred = model.predict(data_test)
		acc_score = round((accuracy_score(labels_test, label_pred)), 4)
		print(f'Number trees: {n_est};  Time: {t_time} ms; Accuracy: {acc_score}')

#Max_depth tests
def max_depth_tests(max_depth_list):
	
	for depth in max_depth_list:
		start_time = time.time()
		model = RandomForestClassifier(criterion="entropy", n_estimators=100, random_state=0, max_depth=depth)
		model.fit(data_train, labels_train)
		end_time = time.time()
		t_time = round(((end_time - start_time) * 1000), 4)
		label_pred = model.predict(data_test)
		acc_score = round((accuracy_score(labels_test, label_pred)), 4)
		print(f'Max depth: {depth}; Time: {t_time} ms; Accuracy: {acc_score}')

#Max bag samples tests
def max_sample_tests(data_percents):
	data_len = len(data_train)
	
	for perc in data_percents:
		start_time = time.time()
		max_sample = int(round(perc*data_len))
		model = RandomForestClassifier(criterion="entropy", n_estimators=100, random_state=0, 
		                               max_depth=5, max_samples=max_sample)
		model.fit(data_train, labels_train)
		end_time = time.time()
		t_time = round(((end_time - start_time) * 1000), 4)
		label_pred = model.predict(data_test)
		acc_score = round((accuracy_score(labels_test, label_pred)), 4)
		print(f'Max bag sample length: {max_sample};  Time: {t_time} ms; Accuracy: {acc_score}')

def sum_of_n_numbers(n):
	# start_time = time.time()
	s = 0
	for i in range(1,n+1):
		start_time = time.time()
		s = s + i
		end_time = time.time()
		print(s, end_time-start_time)
	# end_time = time.time()
	print(s, end_time-start_time)
	# return s,end_time-start_time

# def main():
n_est_list = [1, 10, 100, 500]
max_depth_list = [1, 2, 5, 10, 20]
data_percents = [0.1, 0.3, 0.5, 0.8]

n_est_tests(n_est_list)
max_depth_tests(max_depth_list)
max_sample_tests(data_percents)
# sum_of_n_numbers(4)









