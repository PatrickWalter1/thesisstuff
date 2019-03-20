# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 10:08:29 2018

@author: Patrick
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# data set from jay
path = "2018_Dataset.csv"
dt = pd.read_csv(path)
#remove the timestamps
timestamps = dt['Timestamp']
dt= dt.drop('Timestamp', axis = 1)
print("total dataset:")
print(str(dt.info()))
#print(dt.head())
# need a better work around for NaN rows then just dropping them
# Dropping records with NaN values:
dt =dt.dropna(how='any', axis = 0)
#print(dt.head())
print("Drop NaN records")
print(str(dt.info()))
#print(str(dt.dtypes()))

#divide into testing and training?
#maybe not for regression
train = dt[:-12]
print("Shape of training data:") 
print(train.shape[0], train.shape[1])
test = dt.iloc[-12:]
print("Shape of testing data:") 
print(test.shape[0], test.shape[1])
#split train into attributes and targets
#training attributes are first 281 columns
train_attributes= train.iloc[:,:280]
#targets are not including total cpi
train_groupings_target = train.iloc[:,282:]
#total cpi target
train_CPI_target = train['Food 17']

#split test into attribubtes and targets
#test attributes 
test_attributes= test.iloc[:,:280]
print("Test Attribute matrix:")
print(test_attributes.shape[0], test_attributes.shape[1])
#targets are not including total cpi
test_groupings_target = test.iloc[:,282:]
print(test_groupings_target.shape[0],test_groupings_target.shape[1])
#total cpi target
test_CPI_target = test['Food 17']
test_CPI_target=test_CPI_target.as_matrix()

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(train_attributes, train_CPI_target)


# Make predictions using the testing set
target_CPI_pred = regr.predict(test_attributes)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(test_CPI_target, target_CPI_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_CPI_target, target_CPI_pred))

print(target_CPI_pred)
print("CPI targets")
print(test_CPI_target)


#print("CPI targets")
#rint(test_CPI_target)


"""
regr.fit(train_attributes,test_groupings_target)

# Make predictions using the testing set
target_CPI_pred = regr.predict(test_attributes)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(test_groupings_target, target_CPI_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_groupings_target, target_CPI_pred))
"""
#target_CPI_pred.plot()
x = timestamps[-12:]
# Plot outputs
plt.scatter(x,test_CPI_target,  color='black')
plt.plot(target_CPI_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()



