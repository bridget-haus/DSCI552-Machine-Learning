import numpy as np
import pocket #Import from part1, for data cleaning steps
import pocket_classifier #Import library for part2

def get_inputs(file):
    '''Get inputs in proper format from file'''
    data_list = pocket.read_file(file)
    
    cols_from_left = 3
    X = pocket.get_x(cols_from_left, data_list) #Get xis

    y_index = 4
    Y = pocket.get_y(y_index, data_list)
    y = []
    for yi in Y:
        y.append(yi[0])
    y = np.array(y) #Get yis
    
    #num_xis, num_dims = X.shape
    X_list = X.tolist()
    y_list = y.tolist()
    y_vals = tuple(set(y_list))
    return X_list, y_list, y_vals

def implement_library(X_list, y_list, y_vals, max_iters):
    '''Use the pocket_classifier module to train perceptron.
    X_list - list of rows, list of lists
    y_list - list of row labels, list
    y_vals - set of discrete y labels, tuple
    max_iters - max amount of iterations'''
    num_xis = len(X_list)
    num_dims = len(X_list[0])
    
    pocket_obj = pocket_classifier.PocketClassifier(num_dims, y_vals)
    pocket_obj.train(X_list,y_list, max_iters)
    
    #Get last iteration's weights
    W_final = pocket_obj.__dict__['weights'].tolist() 
    
    #Get misclasses per iteration
    misclassify_record = pocket_obj.__dict__['misclassify_record'] 
    iter_best_incorrect_predictions = list(enumerate(misclassify_record))

    #Get accuracy per iteration
    iter_best_accuracies = [] 
    for i in iter_best_incorrect_predictions:
        accuracy = (i[0], ((num_xis-i[1]) / num_xis))
        iter_best_accuracies.append(accuracy)

    #Get last iteration's accuracy
    accuracy_final = iter_best_accuracies[-1][1] 
    return W_final, iter_best_incorrect_predictions, iter_best_accuracies, accuracy_final

def main():
    file = 'classification.txt'
    X_list, y_list, y_vals = get_inputs(file)

    max_iters = 7000
    W_final, iter_best_incorrect_predictions, iter_best_accuracies, accuracy_final = implement_library(X_list, y_list, y_vals, max_iters)
    print('Final weights:' + '\n', W_final)
    print('Final accuracy:' + '\n', accuracy_final)

    pocket.plot_performance(iter_best_incorrect_predictions, iter_best_accuracies)

if __name__ == "__main__":
    main()