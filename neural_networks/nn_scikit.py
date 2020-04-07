#GOAL: Implement a feed-forward neural network in scikit-learn
from PIL import Image

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

def normalize_dataset(file):
    '''iterate through training/test list and create dataset of pixels
    input: file - a .txt file containing a list of image files downgesture_train.list.txt
    output: dataset - list of image pixels and their associated label, list of lists'''
    with open(file, 'r') as train_data:
        lines = train_data.read().splitlines()
        dataset = []
        for line in lines:
            img = Image.open(line)
            grey_img = img.convert(mode='L')
            pix_val = list(grey_img.getdata())
            # normalize pixels to values between 0 and 1
            for i in range(len(pix_val)):
                pix_val[i] = pix_val[i]/255
            # append label to pixel dataset
            if 'down' in line:
                pix_val.append(1)
            else:
                pix_val.append(0)
            dataset.append(pix_val)
    return dataset

def split_xy(dataset):
    '''Re-format the X and Y training values
    dataset - output from normalize_dataset'''
    X_train = []
    Y_train = []
    for row in dataset:
        X_train.append(row[:-1])
        Y_train.append(row[-1])
    return X_train, Y_train

def main():
    #Get training data
    train_data = normalize_dataset('downgesture_train.list.txt')
    X_train, Y_train = split_xy(train_data)
    
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
    mlp.fit(X_train,Y_train)
    
    #Get testing data
    test_data = normalize_dataset('downgesture_test.list.txt')
    X_test, Y_test = split_xy(test_data)
    
    #Use model to get predictions on test images
    predictions = mlp.predict(X_test)

    #Get prediction stats
    print(classification_report(Y_test,predictions))
    print(confusion_matrix(Y_test,predictions))

if __name__ == "__main__":
    main()
