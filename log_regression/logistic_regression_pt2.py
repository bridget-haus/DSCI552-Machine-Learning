from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
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
    X_data = np.mat(np.array(X_data).astype(float))
    Y_data = np.mat(np.array(Y_data).astype(float)).T

    X_train, X_test, y_train, y_test = train_test_split(X_data, 
       Y_data, test_size=0.05, 
        random_state=101)
    
    logmodel = LogisticRegression(fit_intercept=False)
    logmodel.fit(X_train, y_train)
    
    predictions = logmodel.predict(X_test)
    coef = logmodel.coef_[0]
    print(coef)
    print(classification_report(y_test,predictions))