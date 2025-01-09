# -*- coding: utf-8 -*-
"""Diabetes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SG_R4XWc_LdoJH-lvmrk6MPEhBHNWEvd

Importing Important Depedencies
"""

import numpy as np # To make numpy arrays
import pandas as pd # Used to create data frames (tables)
from sklearn.preprocessing import StandardScaler # To standardize the data
from sklearn.model_selection import train_test_split # To split the data into
# training and testing

from sklearn import svm # Importing Support Vector Machine model
from sklearn.metrics import accuracy_score

"""Data Collection and Analysis"""

# We can probably change the data or switch to a different kind of data.
# this is from a tutorial just to see how well the model works.

# Loading diabetes.csv through panda dataframe

df = pd.read_csv('diabetes.csv')

# Printing first 5 rows of dataset
# df.head()

# df.describe()

# df['Pregnancies'].value_counts()

# df['Outcome'].value_counts()

# Separating data and labels
X = df.drop(columns='Outcome', axis=1) # column of outcome is dropped. rest is assigned
Y = df['Outcome'] # Only outcome is stored corresponding to df's indexes

# print(X)

# Standardizing data now since all values have differenct ranges.
scaler = StandardScaler()

standardized_data = scaler.fit_transform(X) # Fitting and transforming X into
# variable standardized_data
# print(standardized_data)

# Ranges are similar across different columns
X = standardized_data
Y = df['Outcome']

# print(Y)

# Splitting data into Training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)
# 4 variables here, X being split in train and test
# 0.2 = 20% of data (how much of data is test data)
# Stratify is to ensure that the X_train, X_test have the same proportion (diabestic, non-diabetic) as data
# So it will eliminate all non-diabetic data going to train and it being tested on all diabetic people
# Random state is to replicate the same splitting for testing purposes ig

# print(X.shape, X_train.shape, X_test.shape)

# 20% went to X_test rest 80% is with X_train (i.e. more date to train which is good)

# TRAINING THE MODEL
# Loading support vector machine
classifier = svm.SVC(kernel='linear')

# Training SVM
classifier.fit(X_train, Y_train)

# Evaluating the model
# Accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction, Y_train)
# print("Accuracy score of the training data: ", training_accuracy)

X_test_prediction = classifier.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy score of the test data: ", test_accuracy)

# We have our model ready now we just need a predictive system

# input_data = (2, 100, 50, 23, 80, 22.03, 0.3, 45) # hopefully 0
input_data = (3, 100, 40, 23, 80, 22.03, 0.3, 45) # hopefully 0

# Change it to numpy array

input_data = np.asarray(input_data)

# Re-shaping the array (as model is expecting 768 values rn)

input_data = input_data.reshape(1,-1)

# Standardizing the data

input_data = scaler.transform(input_data)

print(input_data)

prediction = classifier.predict(input_data)
print(prediction)

