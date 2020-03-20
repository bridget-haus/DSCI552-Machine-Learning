import sys 
import numpy as np

def sigmoid(x):
    return -1/(1+np.exp(x))

def gradient_descent(sig,X_data,Y_data):
    gradient = np.multiply(sig, np.multiply(X_data,Y_data))
    return np.sum(gradient, axis = 0)

def get_weights(weights, gradient, learning_rate):
    eta = learning_rate
    W = weights - eta * gradient
    return W

def error(countcorrect,len_vals):
    error = round(1 - int(countcorrect) / len_vals,4)
    return error

filename = 'C:\\Users\Peter\\Desktop\\Classes\\INF 552\\Homework 4\\classification.txt'
X_data = []
Y_data = []
eta = -.05 #learning late
count_correct = 0
count_incorrect = 0
with open(filename, "r") as file:
    content = file.read().splitlines()
    for line in content: #Split the data into attributes and classifiers 
        temp = [1]
        for val in line.split(",")[:3]:
            temp.append(val)
        X_data.append(temp)
        Y_data.append(line.split(",")[4])
        
    weights = [] #Randomly assign weights to each of the variables
    for i in range(len(X_data[0])):
        weights.append(np.random.uniform(0,1))
    weights = np.mat(np.array(weights))

    X_data = np.mat(np.array(X_data).astype(float))
    Y_data = np.mat(np.array(Y_data).astype(float)).T
    ywx = np.multiply((X_data * weights.T), Y_data)
    
    for i in range(7000): #iterate recursively 7000 times
        ywx = np.multiply((X_data * weights.T), Y_data)
        gradient = gradient_descent(sigmoid(ywx),X_data,Y_data) * -1/len(Y_data) #*-1/N
        weights = get_weights(weights,gradient,eta)
    
    print(weights)
    
    prediction = np.dot(X_data, weights.T)
    for i in range(len(prediction)):
        if prediction[i]>0:
            my_predict = 1
        else:
            my_predict = -1
            
        if Y_data[i] == my_predict:
            count_correct += 1
        else:
            count_incorrect += 1
    
    total_error = error(count_correct,len(Y_data))
    print(f'error of: {total_error}')