import numpy as np #For matrix and linear algebra
import matplotlib.pyplot as plt #For visualization
from mpl_toolkits.mplot3d import Axes3D 

def read_file(file):
    '''Read the classification.txt file. Output a list of lists.'''
    fd = open(file, "r")
    rows = fd.read().splitlines()
    data_list = []
    for row in rows:
        row2 = row.replace("+", "")
        row_list = row2.split(",")
        row_list_nums = []
        for item in row_list:
            val = float(item)
            if val == 1.0:
                val = int(val)
            elif val == -1.0:
                val = int(val)
            row_list_nums.append(val)
        data_list.append(row_list_nums)
    return data_list

def get_x(cols_from_left, data_list):
    '''Extract x columns from data_list.
    cols_from_left - the number of cols from left you want, int
    data_list - output from read_file(), list of lists'''
    X_list = []
    for row in data_list:
        X_row = row[:cols_from_left]
        X_list.append(X_row)
    X = np.array(X_list)
    return X

def get_y(y_index, data_list):
    '''Extract y columns from data_list.
    y_index - the index of desired y column.'''
    Y_list = []
    for row in data_list:
        Y_row = [row[y_index]]
        Y_list.append(Y_row)
    Y = np.array(Y_list)
    return Y

def plot_3d_data(X, Y):
    ''' Plots 3d data with binary Y-labels. +1s are blue, -1s are red
    X - numpy matrix of xis with 3 dims
    Y - numpy array of yis labelled +1 or -1'''
    featA = X.transpose().tolist()[0]
    featB = X.transpose().tolist()[1]
    featC = X.transpose().tolist()[2]
    
    y_list = []
    for yi in Y:
        y_list.append(yi[0])
        
    featA_tp = []
    featB_tp = []
    featC_tp = []
    y_tp = []
    featA_tn = []
    featB_tn = []
    featC_tn = []
    y_tn = []

    for xiA, xiB, xiC, yi in zip(featA, featB, featC, y_list):
        if yi == 1:
            featA_tp.append(xiA)
            featB_tp.append(xiB)
            featC_tp.append(xiC)
            y_tp.append(yi)
        elif yi == -1:
            featA_tn.append(xiA)
            featB_tn.append(xiB)
            featC_tn.append(xiC)
            y_tn.append(yi)

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='3d')

    for xi,yi in zip(X, Y):
        if yi[0] == 1:
            ax.scatter(xi[0], xi[1], xi[2], c='b', marker='o')
        elif yi[0] == -1:
            ax.scatter(xi[0], xi[1], xi[2], c='r', marker='^')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def sign_function(Wxi):
    '''Assigns +1 if Wxi is greater than 0, and -1 if Wxi is less than 0.
    Wxi - dot product of W and xi, float.'''
    if Wxi < 0:
        return -1
    else:
        return 1

def initialize_params(X):
    '''Add bias column to xis and initialize weights'''
    num_xis, num_dims = X.shape
    # Add a bias (x0 = 1) for each datapoint, xi's are horizontal
    X_anchor = np.hstack((np.ones((num_xis, 1)), X) )
    
    # Initialize weights between 0 and 1, W is vertical
    W = np.random.rand(num_dims+1).reshape((-1,1))
    return X_anchor, W

def pocket(X_anchor, Y, W, learning_rate, max_iters):
    '''Given and initialized X and W, implement perceptron learning to optimize W
    X_anchor - rows of datapoints, with anchor: x0 = 1
    Y - column of labels
    W - initialized weights of hyperplane
    learning_rate - constant between 0 and 1, used in gradient descent step
    max_iters - max iterations through the dataset, before stopping'''
    num_xis, num_dims = X_anchor.shape
    row_misclassified = True
    iter_counter = 0
    
    iter_misclasses = []
    iter_accuracies = []
    
    #For pocket algorithm
    iter_best_incorrect_predictions = []
    iter_best_accuracies = []
    
    best_W = np.zeros(num_dims)
    best_incorrect_predictions = num_xis #Assume when starting, that all data is misclassified
    best_accuracy = 0
    
    while row_misclassified:
        #Start with no misclassification, then update upon misclassification
        row_misclassified = False

        correct_predictions = 0
        incorrect_predictions = 0

        for index, x in enumerate(X_anchor):
            Wxi = np.dot(x.reshape((1,-1)), W)
            if sign_function(Wxi) != Y[index]: # if sample misclassified (i.e. Wxi != yi)
                row_misclassified = True
                incorrect_predictions += 1
                W = W + (Y[index] * learning_rate * x.reshape((-1,1))) #Update W in appropriate direction of misclassified xi
            else:
                correct_predictions += 1
        iter_counter += 1

        #Get the num_misclasses per iteration
        iter_misclass = (iter_counter, incorrect_predictions)
        iter_misclasses.append(iter_misclass)

        #Get the accuracy rate per iteration
        accuracy = round((correct_predictions/num_xis), 5)
        iter_accuracy = (iter_counter, accuracy)
        iter_accuracies.append(iter_accuracy)
        
        #Pocket algorithm
        if incorrect_predictions < best_incorrect_predictions:
            best_W = W
            best_incorrect_predictions = incorrect_predictions
            best_accuracy = accuracy
            
        iter_best_incorrect_pred = (iter_counter, best_incorrect_predictions)
        iter_best_incorrect_predictions.append(iter_best_incorrect_pred)
        
        iter_best_acc = (iter_counter, best_accuracy)
        iter_best_accuracies.append(iter_best_acc)

        #Break from while loop if max_iters is reached
        if iter_counter >= max_iters:
            break
    return W, best_W, iter_misclasses, iter_accuracies, iter_best_incorrect_predictions, iter_best_accuracies

#GOAL: Plot a dual y-axis graph of misclasses and accuracy
def plot_performance(iter_misclasses, iter_accuracies):
    '''Plot a dual y-axis graph of misclasses and accuracy for iterations'''
    iters = []
    misclasses = []
    accuracies = []
    for i in zip(iter_misclasses, iter_accuracies):
        iters.append(i[0][0])
        misclasses.append(i[0][1])
        accuracies.append(i[1][1])
    iters = np.asarray(iters) #x
    misclasses = np.asarray(misclasses) #y1
    accuracies = np.asarray(accuracies) #y2

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('misclasses', color=color)
    ax1.plot(iters, misclasses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)
    ax2.plot(iters, accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

def main():
    file = 'classification.txt'
    data_list = read_file(file)

    cols_from_left = 3
    X = get_x(cols_from_left, data_list)

    y_index = 4
    Y = get_y(y_index, data_list)
    
    plot_3d_data(X, Y)

    X_anchor, W = initialize_params(X)

    learning_rate = 1
    max_iters = 7000
    W, best_W, iter_misclasses, iter_accuracies, iter_best_incorrect_predictions, iter_best_accuracies = pocket(X_anchor, Y, W, learning_rate, max_iters)
    print('Final weights:' + '\n', W)
    print('Final iteration and accuracy:' + '\n', iter_accuracies[-1])
    
    plot_performance(iter_best_incorrect_predictions, iter_best_accuracies)

if __name__ == "__main__":
    main()
