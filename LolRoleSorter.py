# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Loltest.csv')
X = dataset.values


#Econding cateorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [2])
Y = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features = [1])
Z = onehotencoder.fit_transform(X).toarray()

X = Y[:,:-2]
Y = Z[:,:-2]
count = 0
i,j = X.shape

while i>count:

    X[count][0:6] = X[count][0:6] + Y[count]
    count+=1
    


#avoid dummy trap
X = X[:,1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#fit model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predit
y_pred = regressor.predict(X_test)

#Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = X, values = np.ones((50,1)).astype(int))