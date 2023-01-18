# Programming Part
# Assignment # 3
# Question # 2
# Neural Networks (Recall, Precision, Average, Classification Chart)
# CPSC 4310 Computational Intelligence
# Jacqueline Eshriew
# SPRING 2022

import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("Loading SVHN Datasets, Please wait. . .")

 #load the training mats and the testing mats working!
load_train = sio.loadmat('/Users/minmi/Desktop/Assignment-3-Jackie/train_32x32.mat')
load_test = sio.loadmat('/Users/minmi/Desktop/Assignment-3-Jackie/test_32x32.mat')
load_extra = sio.loadmat('/Users/minmi/Desktop/Assignment-3-Jackie/extra_32x32.mat')
# NOTE: Extra dataset was not used at all! Only X and Y!

print("SVHN Datasets Loaded Sucessfully!")

# X is the training
# Y is the trarget

#train datasets
#Loading in the training sets
print("Loading X Training Set. . .")

X = np.asarray(load_train['X'])
X_train = []

for i in range (X.shape[3]):
    X_train.append(X[:,:,:,i])

X_train = np.asarray(X_train)

# Shaping the X set to fit from 4 to 2 sets
# Because models can only use 2
nsamples, nx, ny, nz = X_train.shape
X_train = X_train.reshape((nsamples,nx*ny*nz))

#Print the new dataset format, 2
print(X_train.shape)

#Loading the Y data sets which is 2 so no reshaping
Y_train = load_train['y']

#sort by numbers 0 to 9 (10)
for i in range(len(Y_train)):
    if Y_train[i]%10 == 0:
        Y_train[i] = 0

#Print the datasets
#Print the dataset format, 2
print(Y_train.shape)


#Printing the Y training sets
print("Loading Y Training Set. . .")

#Loading in the tests (Target Data)
#test datasets
X = np.asarray(load_test['X'])
X_test = []

for i in range (X.shape[3]):
    X_test.append(X[:,:,:,i])

X_test = np.asarray(X_test)

# Shaping the X set to fit from 4 to 2 sets
# Because models can only use 2
nsamples, nx, ny, nz = X_test.shape
X_test = X_test.reshape((nsamples,nx*ny*nz))

#Print the new dataset format, 2
print(X_test.shape)

#Loading the Y data sets which is 2 so no reshaping
Y_test = load_test['y']

#sort by numbers 0 to 9 (10)
for i in range(len(Y_test)):
    if Y_test[i]%10 == 0:
        Y_test[i] = 0

#Print the datasets
#Print the dataset format, 2
print(Y_test.shape)


#Set the Neural Networks as the model
#Add the hidden layers, 2 of them
model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5,5), random_state=1)

#Fit the X and Y training sets into the model to be assesed.
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

#Print Model Accuracy, Precision and Recall
print('Accuracy Score : {0:0.2f}'.format(accuracy_score(Y_test, Y_pred)))

print('Recall Macro Score : {0:0.2f}'.format(recall_score(Y_test, Y_pred, average='macro')))
print('Recall Micro Score : {0:0.2f}'.format(recall_score(Y_test, Y_pred, average='micro')))
print('Recall Weighted Score : {0:0.2f}'.format(recall_score(Y_test, Y_pred, average='weighted')))

print('Precision Macro Score : {0:0.2f}'.format(precision_score(Y_test, Y_pred, average='macro')))
print('Precision Micro Score : {0:0.2f}'.format(precision_score(Y_test, Y_pred, average='micro')))
print('Precision Weighted Score : {0:0.2f}'.format(precision_score(Y_test, Y_pred, average='weighted')))

#Print the full evaluation of the Nerual Network classification
print("\nClassification Report : ")
print(classification_report(Y_test,Y_pred))
