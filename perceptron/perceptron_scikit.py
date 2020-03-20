import numpy as np
from sklearn.linear_model import Perceptron
import perceptron #Import the perceptron.py file from part 1

def get_inputs(file):
    '''Get inputs in proper format from file'''
    data_list = perceptron.read_file(file)
    
    cols_from_left = 3
    X = perceptron.get_x(cols_from_left, data_list) #Get xis

    y_index = 3
    Y = perceptron.get_y(y_index, data_list)
    y = []
    for yi in Y:
        y.append(yi[0])
    y = np.array(y) #Get yis

    X_anchor, W = perceptron.initialize_params(X)
    return X, y

def learn_perceptron(X, y):
    clf = Perceptron(max_iter=50, eta0=.1, tol=None)
    clf.fit(X,y)

    w_intercept = clf.intercept_ #final weight for x0
    w_xis = clf.coef_ #final weights for x1, x2, x3
    accuracy = clf.score(X, y)

    print('w_intercept:', w_intercept)
    print('w_xis:', w_xis)
    print('mean_accuracy:', accuracy)

def main():
    file = 'classification.txt'
    X, y = get_inputs(file)
    learn_perceptron(X, y)

if __name__ == "__main__":
    main()