import scikit_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class_names, feature_names, data_train, data_test, labels_train, labels_test = scikit_tree.split_train_test('data.csv')

#Random state tests

#N_estimators tests
n_est_list = [1, 10, 100, 500]
for n_est in n_est_list:
    model = RandomForestClassifier(criterion="entropy", n_estimators=n_est, random_state=0)
    model.fit(data_train, labels_train)
    label_pred = model.predict(data_test)
    print(f'Number trees: {n_est}, Accuracy of model is: {accuracy_score(labels_test, label_pred)}')

#Max_depth tests
max_depth_list = [1, 2, 5, 10, 20]
for depth in max_depth_list:
    model = RandomForestClassifier(criterion="entropy", n_estimators=100, random_state=0, max_depth=depth)
    model.fit(data_train, labels_train)
    label_pred = model.predict(data_test)
    print(f'Max depth: {depth}, Accuracy: {accuracy_score(labels_test, label_pred)}')

#Max bag samples tests
data_len = len(data_train)
data_percents = [0.1, 0.3, 0.5, 0.8]
for perc in data_percents:
    max_sample = int(round(perc*data_len))
    model = RandomForestClassifier(criterion="entropy", n_estimators=100, random_state=0, 
                                   max_depth=5, max_samples=max_sample)
    model.fit(data_train, labels_train)
    label_pred = model.predict(data_test)
    print(f'Max bag sample length: {max_sample}, Accuracy: {accuracy_score(labels_test, label_pred)}')