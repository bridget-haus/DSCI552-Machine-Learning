import time
import scikit_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class_names, feature_names, data_train, data_test, labels_train, labels_test = scikit_tree.split_train_test('data.csv')

#N_estimators tests
def n_est_tests(n_est_list):
	time_list = []
	acc_list = []
	for n_est in n_est_list:
		model = RandomForestClassifier(criterion="entropy", n_estimators=n_est, random_state=0)
		start_time = time.time()
		model.fit(data_train, labels_train)
		end_time = time.time()
		t_time = round(((end_time - start_time) * 1000), 4)
		time_list.append(t_time)

		label_pred = model.predict(data_test)
		acc_score = round((accuracy_score(labels_test, label_pred)), 4)
		acc_list.append(acc_score)

		print(f'Number trees: {n_est};  Training Time: {t_time} ms; Accuracy: {acc_score}')
	return time_list, acc_list

#Max_depth tests
def max_depth_tests(max_depth_list):
	time_list = []
	acc_list = []
	for depth in max_depth_list:
		model = RandomForestClassifier(criterion="entropy", n_estimators=100, random_state=0, max_depth=depth)
		start_time = time.time()
		model.fit(data_train, labels_train)
		end_time = time.time()
		t_time = round(((end_time - start_time) * 1000), 4)
		time_list.append(t_time)

		label_pred = model.predict(data_test)
		acc_score = round((accuracy_score(labels_test, label_pred)), 4)
		acc_list.append(acc_score)

		print(f'Max depth: {depth}; Training Time: {t_time} ms; Accuracy: {acc_score}')

	return time_list, acc_list

def plot(x, y1, y2, **kwargs):
	fig, ax1 = plt.subplots()

	color = 'tab:red'
	if kwargs.get('xlabel'): #Label the x axis accordingly
		xlabel = kwargs.get('xlabel')
		ax1.set_xlabel(xlabel)

	if xlabel == 'max_depth':
		ax1.set_xticks(range(0, (5+1)*4, 4)[1:])  

	if kwargs.get('scale'): #Convert x to log scale if desired
		scale = kwargs.get('scale')
		ax1.set_xscale(scale, basex=10)

	ax1.set_ylabel('time (ms)', color=color)
	ax1.plot(x, y1, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
	ax2.plot(x, y2, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()

def main():
	# #NUM TREES
	n_est_list = [1, 2, 5, 7, 10, 25, 50, 75, 100, 250, 500, 750, 1000]
	# # n_est_list = range(0, (100+1)*10, 10)[1:]
	
	time_list, acc_list = n_est_tests(n_est_list)
	print('time_list: ', time_list, 'acc_list: ', acc_list)
	plot(n_est_list, time_list, acc_list, xscale='log', xlabel='num_trees')

	#MAX DEPTH
	max_depth_list = range(1,20)
	
	time_list, acc_list = max_depth_tests(max_depth_list)
	print('time_list: ', time_list, 'acc_list: ', acc_list)
	plot(max_depth_list, time_list, acc_list, xlabel='max_depth')

if __name__ == '__main__':
	main()










