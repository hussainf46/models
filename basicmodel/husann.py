# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:58:21 2019

@author: Dell
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
 
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[: , 3:13 ].values
y = dataset.iloc[: , 13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x1 = LabelEncoder()
x[:, 1] = labelencoder_x1.fit_transform(x[:,1])
labelencoder_x2 = LabelEncoder()
x[:,2] = labelencoder_x2.fit_transform(x[:,2])
encoder = OneHotEncoder(categorical_features = [1])
x = encoder.fit_transform(x).toarray()
x =x[:,1:]

from sklearn.model_selection import train_test_split
x_train , x_test , y_train ,y_test = train_test_split(x ,y ,test_size = 0.2 , random_state = 0 ) 

from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
x_train = standardscaler.fit_transform(x_train)
x_test = standardscaler.transform(x_test)


#importing packages  using keras and tenserflow backend 
import keras 
from keras.models import Sequential 
from keras.layers import Dense 

#initializing the ann
classifier = Sequential()
 #adding first hidden layer and input layer 
classifier.add(Dense(output_dim = 6 , init = 'uniform' , activation = 'relu' , input_dim = 11))

#output_dim specifis the njumber of output you want (its basically the number of nodes in the hidden layer 
#normally it is calculated as the average of the number of dependent and in dependent variables )........
#init is used the adjust the weights before each node 
#activation defines which type if algorithm we want to use for ((activation function ))
#input_dim it is used when you want to add the first hidden layer as we have to specify the number of independent 
#variables it should consuder as the input variables 

# adding second hidden a layer taking input from the first hidden layer 
classifier.add(Dense(output_dim = 6 ,activation = 'relu',init = 'uniform' ))
#adding third hidde layer 
classifier.add(Dense(output_dim = 6 , activation = 'relu' ,init = 'uniform'))
#adding the output layer 
classifier.add(Dense(output_dim = 1 , activation = 'sigmoid' , init = 'uniform'))
#compiling of ann
classifier.compile( optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

#optmizer it is used to optmized the weigths i.e it is  used to make weights more accurate 
#loss it is used  for the output so we have to specify that we want which type if the ouput is binary we used (binary_crossentropy)
#and for categorical we used (categorical_crossentropy)
#metrics is used to keep the track of accuracy i.e it is accuracy metrics 

#fitting the ann
classifier.fit(x_train ,y_train , batch_size = 10 , epochs = 100)
#mking the predictions and evaluating the models 
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)



























 


