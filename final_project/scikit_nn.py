import scikit_tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import time

def scikit_nn():
  class_names, feature_names, data_train, data_test, labels_train, labels_test = scikit_tree.split_train_test('data.csv')

  #Initialize model with parameters
  mlp = MLPClassifier(hidden_layer_sizes=(100), 
                     activation = 'logistic',
                     solver = 'sgd',
                     learning_rate = 'constant',
                     learning_rate_init = 0.1,
                     max_iter = 1000,
                     random_state = 0,
                     verbose = False)

  #Fit the model with X_train and Y_train
  start = time.time()
  mlp.fit(data_train,labels_train)
  end = time.time()
  train_time = round(((end - start) * 1000), 4)
  print(f'Training time: {train_time} ms')

  #Use model to get predictions on test images
  start = time.time()
  predictions = mlp.predict(data_test)
  end = time.time()
  test_time = round(((end - start) * 1000), 4)
  print(f'Testing time: {test_time} ms')

  #Get prediction stats
  print(classification_report(labels_test,predictions))
  print(confusion_matrix(labels_test,predictions))

if __name__ == '__main__':
  scikit_nn()