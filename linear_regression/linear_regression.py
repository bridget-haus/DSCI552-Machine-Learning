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
    
    # Solve with matrix computations W_opt = (D*DT)^-1 * D*y
    # D == X_data;  y == Y_data    
    
    X_data = np.mat(np.array(X_data).astype(float))
    Y_data = np.mat(np.array(Y_data).astype(float))

    
    X_new = X_data.T* X_data
    X_new = X_new.I
    Y_new = Y_data*X_data
    Y_new = Y_new.T

    weights = np.dot(X_new,Y_new,out=None).T
    print(weights)