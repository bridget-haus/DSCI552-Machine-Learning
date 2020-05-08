import scikit_tree
import pandas as pd

class_names, feature_names, data_train, data_test, labels_train, labels_test = scikit_tree.split_train_test('data.csv')

