from sklearn.linear_model import LinearRegression
import numpy as np
filename = 'C:\\Users\Peter\\Desktop\\Classes\\INF 552\\Homework 4\\linear-regression.txt'

X_data = []
Y_data = []
with open(filename, "r") as file:
    content = file.read().splitlines()
    for line in content:
        temp = [1]
        for val in line.split(",")[:2]:
            temp.append(val)
        X_data.append(temp)
        Y_data.append(line.split(",")[2])
    
    
    X_data = np.mat(np.array(X_data).astype(float))
    Y_data = np.mat(np.array(Y_data).astype(float)).T

    linear_regression = LinearRegression(fit_intercept=False).fit(X_data, Y_data)
    lr_coef = linear_regression.coef_
    print(lr_coef)