# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 10:08:29 2018

@author: Patrick Walter
"""
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_regression, mutual_info_regression


# data set from jay divided into two files, and august 2017 removed
attributes = "2018_Attributes.csv"
attributes_df = pd.read_csv(attributes)
targets = "2018_Targets.csv"
#targets dataframe contains all groups including Food 17
targets_df = pd.read_csv(targets)

targets_df = targets_df[170:-7]
# -7 to take off months of 2017
attributes_df = attributes_df[170:-7]

timestamps_df = attributes_df['Timestamp']
food17_target_df = targets_df['Food 17']

# remoe the timestamps from both
targets_df = targets_df.drop('Timestamp', axis = 1)
attributes_df = attributes_df.drop('Timestamp', axis = 1)


# drop all attributes that are missing values for now
# revise this later when we predict them.
attributes_df =attributes_df.dropna(how='any', axis = 1)
attributes_df =attributes_df.dropna(how='any', axis = 1)
#feature selection by variance
#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
#attributes_df = sel.fit_transform(attributes_df)
#fEATURE selection with k = 18
attributes_df = SelectKBest(mutual_info_regression, k = 18).fit_transform(attributes_df,food17_target_df)

print("New taregsts dataframe:")
#print(attributes_df.info())
#print(attributes_df.head())
#print(attributes_df.head())

#how much to take off timeframe in months negative
#will be used for predicting each year based on years before
#currently doesnt line up with years since last record is July 2017
# the negative 7 lines it up

eighteenyear = -216 #1999
sixteenyear = -192 #year 2000
seventeenyear = -204
fifteenyear = -180
fourteenyear = -168
thirteenyear = -156
twelveyear = -144
elevenyear = -132
tenyear = -120
nineyear = -108
eightyear = -96
sevenyear = -84
sixyear = - 72
fiveyear = -60 #months
fouryear = -48
threeyear = -36
twoyear = -24
oneyear = -12  # 2016 


#one year divide into testing and training 
#attributes
att_train_oneyear_df = attributes_df[:oneyear]
att_test_oneyear_df = attributes_df[oneyear:]
#target groupings
tar_train_oneyear__df = targets_df[:oneyear]
tar_test_oneyear_df = targets_df[oneyear:]
# food 17 alone
food17_train_oneyear_df =food17_target_df[:oneyear]
food17_test_oneyear_df =food17_target_df[oneyear:]


#2 years ago
#attributes
att_train_twoyear_df = attributes_df[:twoyear]
att_test_twoyear_df = attributes_df[twoyear:oneyear]
#target groupings
tar_train_twoyear__df = targets_df[:twoyear]
tar_test_twoyear_df = targets_df[twoyear:oneyear]
# food 17 alone
food17_train_twoyear_df =food17_target_df[:twoyear]
food17_test_twoyear_df =food17_target_df[twoyear:oneyear]

#3 years ago
#attributes
att_train_threeyear_df = attributes_df[:threeyear]
att_test_threeyear_df = attributes_df[threeyear:twoyear]
#target groupings
tar_train_threeyear__df = targets_df[:threeyear]
tar_test_threeyear_df = targets_df[threeyear:twoyear]
# food 17 alone
food17_train_threeyear_df =food17_target_df[:threeyear]
food17_test_threeyear_df =food17_target_df[threeyear:twoyear]

#4 years ago
#attributes
att_train_fouryear_df = attributes_df[:fouryear]
att_test_fouryear_df = attributes_df[fouryear:threeyear]
#target groupings
tar_train_fouryear__df = targets_df[:fouryear]
tar_test_fouryear_df = targets_df[fouryear:threeyear]
# food 17 alone
food17_train_fouryear_df =food17_target_df[:fouryear]
food17_test_fouryear_df =food17_target_df[fouryear:threeyear]

#5 years ago
#attributes
att_train_fiveyear_df = attributes_df[:fiveyear]
att_test_fiveyear_df = attributes_df[fiveyear:fouryear]
#target groupings
tar_train_fiveyear__df = targets_df[:fiveyear]
tar_test_fiveyear_df = targets_df[fiveyear:fouryear]
# food 17 alone
food17_train_fiveyear_df =food17_target_df[:fiveyear]
food17_test_fiveyear_df =food17_target_df[fiveyear:fouryear]

#6 years ago
#attributes
att_train_6year_df = attributes_df[:sixyear]
att_test_6year_df = attributes_df[sixyear:fiveyear]
#target groupings
tar_train_6year__df = targets_df[:sixyear]
tar_test_6year_df = targets_df[sixyear:fiveyear]
# food 17 alone
food17_train_6year_df =food17_target_df[:sixyear]
food17_test_6year_df =food17_target_df[sixyear:fiveyear]

#7 years ago
#attributes
att_train_7year_df = attributes_df[:sevenyear]
att_test_7year_df = attributes_df[sevenyear:sixyear]
#target groupings
tar_train_7year__df = targets_df[:sevenyear]
tar_test_7year_df = targets_df[sevenyear:sixyear]
# food 17 alone
food17_train_7year_df =food17_target_df[:sevenyear]
food17_test_7year_df =food17_target_df[sevenyear:sixyear]


#8 years ago
#attributes
att_train_8year_df = attributes_df[:eightyear]
att_test_8year_df = attributes_df[eightyear:sevenyear]
#target groupings
tar_train_8year__df = targets_df[:eightyear]
tar_test_8year_df = targets_df[eightyear:sevenyear]
# food 17 alone
food17_train_8year_df =food17_target_df[:eightyear]
food17_test_8year_df =food17_target_df[eightyear:sevenyear]

#9 years ago
#attributes
att_train_9year_df = attributes_df[:nineyear]
att_test_9year_df = attributes_df[nineyear:eightyear]
#target groupings
tar_train_9year__df = targets_df[:nineyear]
tar_test_9year_df = targets_df[nineyear:eightyear]
# food 17 alone
food17_train_9year_df =food17_target_df[:nineyear]
food17_test_9year_df =food17_target_df[nineyear:eightyear]

#10 years ago
#attributes
att_train_10year_df = attributes_df[:tenyear]
att_test_10year_df = attributes_df[tenyear:nineyear]
#target groupings
tar_train_10year__df = targets_df[:tenyear]
tar_test_10year_df = targets_df[tenyear:nineyear]
# food 17 alone
food17_train_10year_df =food17_target_df[:tenyear]
food17_test_10year_df =food17_target_df[tenyear:nineyear]

#11 years ago
#attributes
att_train_11year_df = attributes_df[:elevenyear]
att_test_11year_df = attributes_df[elevenyear:tenyear]
#target groupings
tar_train_11year__df = targets_df[:elevenyear]
tar_test_11year_df = targets_df[elevenyear:tenyear]
# food 17 alone
food17_train_11year_df =food17_target_df[:elevenyear]
food17_test_11year_df =food17_target_df[elevenyear:tenyear]

#12 years ago
#attributes
att_train_12year_df = attributes_df[:twelveyear]
att_test_12year_df = attributes_df[twelveyear:elevenyear]
#target groupings
tar_train_12year__df = targets_df[:twelveyear]
tar_test_12year_df = targets_df[twelveyear:elevenyear]
# food 17 alone
food17_train_12year_df =food17_target_df[:twelveyear]
food17_test_12year_df =food17_target_df[twelveyear:elevenyear]

#13 years ago
#attributes
att_train_13year_df = attributes_df[:thirteenyear]
att_test_13year_df = attributes_df[thirteenyear:twelveyear]
#target groupings
tar_train_13year__df = targets_df[:thirteenyear]
tar_test_13year_df = targets_df[thirteenyear:twelveyear]
# food 17 alone
food17_train_13year_df =food17_target_df[:thirteenyear]
food17_test_13year_df =food17_target_df[thirteenyear:twelveyear]

#14 years ago
#attributes
att_train_14year_df = attributes_df[:fourteenyear]
att_test_14year_df = attributes_df[fourteenyear:thirteenyear]
#target groupings
tar_train_14year__df = targets_df[:fourteenyear]
tar_test_14year_df = targets_df[fourteenyear:thirteenyear]
# food 17 alone
food17_train_14year_df =food17_target_df[:fourteenyear]
food17_test_14year_df =food17_target_df[fourteenyear:thirteenyear]

#15 years ago
#attributes
att_train_15year_df = attributes_df[:fifteenyear]
att_test_15year_df = attributes_df[fifteenyear:fourteenyear]
#target groupings
tar_train_15year__df = targets_df[:fifteenyear]
tar_test_15year_df = targets_df[fifteenyear:fourteenyear]
# food 17 alone
food17_train_15year_df =food17_target_df[:fifteenyear]
food17_test_15year_df =food17_target_df[fifteenyear:fourteenyear]

#16 years ago
#attributes
att_train_16year_df = attributes_df[:sixteenyear]
att_test_16year_df = attributes_df[sixteenyear:fifteenyear]
#target groupings
tar_train_16year__df = targets_df[:sixteenyear]
tar_test_16year_df = targets_df[sixteenyear:fifteenyear]
# food 17 alone
food17_train_16year_df =food17_target_df[:sixteenyear]
food17_test_16year_df =food17_target_df[sixteenyear:fifteenyear]

#17 years ago
#attributes
att_train_17year_df = attributes_df[:seventeenyear]
att_test_17year_df = attributes_df[seventeenyear:sixteenyear]
#target groupings
tar_train_17year__df = targets_df[:seventeenyear]
tar_test_17year_df = targets_df[seventeenyear:sixteenyear]
# food 17 alone
food17_train_17year_df =food17_target_df[:seventeenyear]
food17_test_17year_df =food17_target_df[seventeenyear:sixteenyear]



print("one year back train and test shapes:")
print(att_train_oneyear_df.shape[0], att_train_oneyear_df.shape[1])
print(att_test_oneyear_df.shape[0], att_test_oneyear_df.shape[1])
print("2 year back train and test shapes:")
print(att_train_twoyear_df.shape[0], att_train_oneyear_df.shape[1])
print(att_test_twoyear_df.shape[0], att_test_oneyear_df.shape[1])
print("3 year back train and test shapes:")
print(att_train_threeyear_df.shape[0], att_train_oneyear_df.shape[1])
print(att_test_threeyear_df.shape[0], att_test_oneyear_df.shape[1])
print("4 year back train and test shapes:")
print(att_train_fouryear_df.shape[0], att_train_oneyear_df.shape[1])
print(att_test_fouryear_df.shape[0], att_test_oneyear_df.shape[1])
print("5 year back train and test shapes:")
print(att_train_fiveyear_df.shape[0], att_train_oneyear_df.shape[1])
print(att_test_fiveyear_df.shape[0], att_test_oneyear_df.shape[1])



#LAST 12 MONTHS!
#code below modified version of:
#http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# Create linear regression object for one year
regr_oneyear = linear_model.LinearRegression()
# Train the model using the training sets
regr_oneyear.fit(att_train_oneyear_df, food17_train_oneyear_df)
# Make predictions using the testing set
food17_pred_oneyear = regr_oneyear.predict(att_test_oneyear_df)
# The coefficients
print("Model 1: training data 1999-2016:")
print('Coefficients: \n', regr_oneyear.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_oneyear_df, food17_pred_oneyear))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_oneyear_df, food17_pred_oneyear))



#TWO YEARS BACK 12 MONTHS!
regr_twoyear = linear_model.LinearRegression()
# Train the model using the training sets
regr_twoyear.fit(att_train_twoyear_df, food17_train_twoyear_df)
# Make predictions using the testing set
food17_pred_twoyear = regr_twoyear.predict(att_test_twoyear_df)
# The coefficients
print("Model 2: training data 1999-2015:")
print('Coefficients: \n', regr_twoyear.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_twoyear_df, food17_pred_twoyear))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_twoyear_df, food17_pred_twoyear))


#THREE YEARS BACK 12 MONTHS!
regr_threeyear = linear_model.LinearRegression()
# Train the model using the training sets
regr_threeyear.fit(att_train_threeyear_df, food17_train_threeyear_df)
# Make predictions using the testing set
food17_pred_threeyear = regr_threeyear.predict(att_test_threeyear_df)
# The coefficients
print("Model 3: training data 1999-2014:")
print('Coefficients: \n', regr_threeyear.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_threeyear_df, food17_pred_threeyear))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_threeyear_df, food17_pred_threeyear))


#FOUR YEARS BACK 12 MONTHS!
regr_fouryear = linear_model.LinearRegression()
# Train the model using the training sets
regr_fouryear.fit(att_train_fouryear_df, food17_train_fouryear_df)
# Make predictions using the testing set
food17_pred_fouryear = regr_fouryear.predict(att_test_fouryear_df)
# The coefficients
print("Model 4: training data 1999-2013:")
print('Coefficients: \n', regr_fouryear.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_fouryear_df, food17_pred_fouryear))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_fouryear_df, food17_pred_fouryear))

#FIVE YEARS BACK 12 MONTHS!
regr_fiveyear = linear_model.LinearRegression()
# Train the model using the training sets
regr_fiveyear.fit(att_train_fiveyear_df, food17_train_fiveyear_df)
# Make predictions using the testing set
food17_pred_fiveyear = regr_fiveyear.predict(att_test_fiveyear_df)
# The coefficients
print("Model 5: traning data 1999-2012:")
print('Coefficients: \n', regr_fiveyear.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_fiveyear_df, food17_pred_fiveyear))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_fiveyear_df, food17_pred_fiveyear))

#6 YEARS BACK 12 MONTHS!
regr_6year = linear_model.LinearRegression()
# Train the model using the training sets
regr_6year.fit(att_train_6year_df, food17_train_6year_df)
# Make predictions using the testing set
food17_pred_6year = regr_6year.predict(att_test_6year_df)
# The coefficients
print("Model 6: training data 1999-2011:")
print('Coefficients: \n', regr_6year.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_6year_df, food17_pred_6year))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_6year_df, food17_pred_6year))

#7 YEARS BACK 12 MONTHS!
regr_7year = linear_model.LinearRegression()
# Train the model using the training sets
regr_7year.fit(att_train_7year_df, food17_train_7year_df)
# Make predictions using the testing set
food17_pred_7year = regr_7year.predict(att_test_7year_df)
# The coefficients
print("Model 7: training data 1999-2010:")
print('Coefficients: \n', regr_7year.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_7year_df, food17_pred_7year))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_7year_df, food17_pred_7year))

#8 YEARS BACK 12 MONTHS!
regr_8year = linear_model.LinearRegression()
# Train the model using the training sets
regr_8year.fit(att_train_8year_df, food17_train_8year_df)
# Make predictions using the testing set
food17_pred_8year = regr_8year.predict(att_test_8year_df)
# The coefficients
print("Model 8: training data 1999-2009:")
print('Coefficients: \n', regr_8year.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_8year_df, food17_pred_8year))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_8year_df, food17_pred_8year))

#9 YEARS BACK 12 MONTHS!
regr_9year = linear_model.LinearRegression()
# Train the model using the training sets
regr_9year.fit(att_train_9year_df, food17_train_9year_df)
# Make predictions using the testing set
food17_pred_9year = regr_9year.predict(att_test_9year_df)
# The coefficients
print("|Model 9: training data 1999-2008:")
print('Coefficients: \n', regr_9year.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_9year_df, food17_pred_9year))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_9year_df, food17_pred_9year))

#10 YEARS BACK 12 MONTHS!
regr_10year = linear_model.LinearRegression()
# Train the model using the training sets
regr_10year.fit(att_train_10year_df, food17_train_10year_df)
# Make predictions using the testing set
food17_pred_10year = regr_10year.predict(att_test_10year_df)
# The coefficients
print("Model 10: training data 1999-2007:")
print('Coefficients: \n', regr_10year.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_10year_df, food17_pred_10year))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_10year_df, food17_pred_10year))


#11 YEARS BACK 12 MONTHS!
regr_11year = linear_model.LinearRegression()
# Train the model using the training sets
regr_11year.fit(att_train_11year_df, food17_train_11year_df)
# Make predictions using the testing set
food17_pred_11year = regr_11year.predict(att_test_11year_df)
# The coefficients
print("Model 11: training data 1999-2006:")
print('Coefficients: \n', regr_11year.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_11year_df, food17_pred_11year))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_11year_df, food17_pred_11year))

#11 YEARS BACK 12 MONTHS!
regr_12year = linear_model.LinearRegression()
# Train the model using the training sets
regr_12year.fit(att_train_12year_df, food17_train_12year_df)
# Make predictions using the testing set
food17_pred_12year = regr_12year.predict(att_test_12year_df)
# The coefficients
print("Model 12: training data 1999-2005:")
print('Coefficients: \n', regr_12year.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_12year_df, food17_pred_12year))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_12year_df, food17_pred_12year))

#11 YEARS BACK 12 MONTHS!
regr_13year = linear_model.LinearRegression()
# Train the model using the training sets
regr_13year.fit(att_train_13year_df, food17_train_13year_df)
# Make predictions using the testing set
food17_pred_13year = regr_13year.predict(att_test_13year_df)
# The coefficients
print("Model 13: training data 1999-2004:")
print('Coefficients: \n', regr_13year.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_13year_df, food17_pred_13year))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_13year_df, food17_pred_13year))

#14 YEARS BACK 12 MONTHS!
regr_14year = linear_model.LinearRegression()
# Train the model using the training sets
regr_14year.fit(att_train_14year_df, food17_train_14year_df)
# Make predictions using the testing set
food17_pred_14year = regr_14year.predict(att_test_14year_df)
# The coefficients
print("Model 14: training data 1999-2003:")
print('Coefficients: \n', regr_14year.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_14year_df, food17_pred_14year))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_14year_df, food17_pred_14year))

#15 YEARS BACK 12 MONTHS!
regr_15year = linear_model.LinearRegression()
# Train the model using the training sets
regr_15year.fit(att_train_15year_df, food17_train_15year_df)
# Make predictions using the testing set
food17_pred_15year = regr_15year.predict(att_test_15year_df)
# The coefficients
print("Model 15: training data 1999-2002:")
print('Coefficients: \n', regr_15year.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_15year_df, food17_pred_15year))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_15year_df, food17_pred_15year))

#16 YEARS BACK 12 MONTHS!
regr_16year = linear_model.LinearRegression()
# Train the model using the training sets
regr_16year.fit(att_train_16year_df, food17_train_16year_df)
# Make predictions using the testing set
food17_pred_16year = regr_16year.predict(att_test_16year_df)
# The coefficients
print("Model 16: training data 1999-2001:")
print('Coefficients: \n', regr_16year.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_16year_df, food17_pred_16year))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_16year_df, food17_pred_16year))

#17 YEARS BACK 12 MONTHS!
regr_17year = linear_model.LinearRegression()
# Train the model using the training sets
regr_17year.fit(att_train_17year_df, food17_train_17year_df)
# Make predictions using the testing set
food17_pred_17year = regr_17year.predict(att_test_17year_df)
# The coefficients
print("Model 17: training data 1999-2000:")
print('Coefficients: \n', regr_17year.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_17year_df, food17_pred_17year))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_17year_df, food17_pred_17year))



# BAGGGING STARTS HERE:
 # used the 17 models on the same test set to make predictions   
food17_pred_17year = regr_17year.predict(att_test_oneyear_df)
food17_pred_16year = regr_16year.predict(att_test_oneyear_df)
food17_pred_15year = regr_15year.predict(att_test_oneyear_df)
food17_pred_14year = regr_14year.predict(att_test_oneyear_df)
food17_pred_13year = regr_13year.predict(att_test_oneyear_df)
food17_pred_12year = regr_12year.predict(att_test_oneyear_df)
food17_pred_11year = regr_11year.predict(att_test_oneyear_df)
food17_pred_10year = regr_10year.predict(att_test_oneyear_df)
food17_pred_9year = regr_9year.predict(att_test_oneyear_df)
food17_pred_8year = regr_8year.predict(att_test_oneyear_df)
food17_pred_7year = regr_7year.predict(att_test_oneyear_df)
food17_pred_6year = regr_6year.predict(att_test_oneyear_df)
food17_pred_fiveyear = regr_fiveyear.predict(att_test_oneyear_df)
food17_pred_fouryear = regr_fouryear.predict(att_test_oneyear_df)
food17_pred_threeyear = regr_threeyear.predict(att_test_oneyear_df)
food17_pred_twoyear = regr_twoyear.predict(att_test_oneyear_df)
food17_pred_oneyear = regr_oneyear.predict(att_test_oneyear_df)

# average the 17 predictions
food17_bagged_allyears = np.mean([
        food17_pred_17year,
        food17_pred_16year,
        food17_pred_15year,
        food17_pred_14year,
        food17_pred_13year,
        food17_pred_12year,
        food17_pred_11year,
        food17_pred_10year,
        food17_pred_9year,
        food17_pred_8year,
        food17_pred_7year,
        food17_pred_6year,
        food17_pred_fiveyear,
        food17_pred_fouryear,
        food17_pred_threeyear,
        food17_pred_twoyear,
        food17_pred_oneyear
        ], axis = 0)
 # print the predictions
print(food17_pred_17year,
        food17_pred_16year,
        food17_pred_15year,
        food17_pred_14year,
        food17_pred_13year,
        food17_pred_12year,
        food17_pred_11year,
        food17_pred_10year,
        food17_pred_9year,
        food17_pred_8year,
        food17_pred_7year,
        food17_pred_6year,
        food17_pred_fiveyear,
        food17_pred_fouryear,
        food17_pred_threeyear,
        food17_pred_twoyear,
        food17_pred_oneyear)

print("Bagged Results average")
print(food17_bagged_allyears)  
print("bootstraped aggregated 17 years for last year: ")    
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_oneyear_df, food17_bagged_allyears))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_oneyear_df, food17_bagged_allyears))

print(food17_bagged_allyears)



#tested on year 2000
    
food17_pred_17year = regr_17year.predict(att_test_17year_df)
food17_pred_16year = regr_16year.predict(att_test_17year_df)
food17_pred_15year = regr_15year.predict(att_test_17year_df)
food17_pred_14year = regr_14year.predict(att_test_17year_df)
food17_pred_13year = regr_13year.predict(att_test_17year_df)
food17_pred_12year = regr_12year.predict(att_test_17year_df)
food17_pred_11year = regr_11year.predict(att_test_17year_df)
food17_pred_10year = regr_10year.predict(att_test_17year_df)
food17_pred_9year = regr_9year.predict(att_test_17year_df)
food17_pred_8year = regr_8year.predict(att_test_17year_df)
food17_pred_7year = regr_7year.predict(att_test_17year_df)
food17_pred_6year = regr_6year.predict(att_test_17year_df)
food17_pred_fiveyear = regr_fiveyear.predict(att_test_17year_df)
food17_pred_fouryear = regr_fouryear.predict(att_test_17year_df)
food17_pred_threeyear = regr_threeyear.predict(att_test_17year_df)
food17_pred_twoyear = regr_twoyear.predict(att_test_17year_df)
food17_pred_oneyear = regr_oneyear.predict(att_test_17year_df)

food17_bagged_allyears = np.mean([
        food17_pred_17year,
        food17_pred_16year,
        food17_pred_15year,
        food17_pred_14year,
        food17_pred_13year,
        food17_pred_12year,
        food17_pred_11year,
        food17_pred_10year,
        food17_pred_9year,
        food17_pred_8year,
        food17_pred_7year,
        food17_pred_6year,
        food17_pred_fiveyear,
        food17_pred_fouryear,
        food17_pred_threeyear,
        food17_pred_twoyear,
        food17_pred_oneyear
        ], axis = 0)
    
print("bootstraped aggregated 17 years for 17 years ago: ")    
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_17year_df, food17_bagged_allyears))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_17year_df, food17_bagged_allyears))

print("Bagged all years average:")
print(food17_bagged_allyears)


#BAGGGING STARTS HERE: weighted averaging with weights = 1/2^i
#                       where i = # of years back
    
food17_pred_17year = regr_17year.predict(att_test_oneyear_df)
food17_pred_16year = regr_16year.predict(att_test_oneyear_df)
food17_pred_15year = regr_15year.predict(att_test_oneyear_df)
food17_pred_14year = regr_14year.predict(att_test_oneyear_df)
food17_pred_13year = regr_13year.predict(att_test_oneyear_df)
food17_pred_12year = regr_12year.predict(att_test_oneyear_df)
food17_pred_11year = regr_11year.predict(att_test_oneyear_df)
food17_pred_10year = regr_10year.predict(att_test_oneyear_df)
food17_pred_9year = regr_9year.predict(att_test_oneyear_df)
food17_pred_8year = regr_8year.predict(att_test_oneyear_df)
food17_pred_7year = regr_7year.predict(att_test_oneyear_df)
food17_pred_6year = regr_6year.predict(att_test_oneyear_df)
food17_pred_fiveyear = regr_fiveyear.predict(att_test_oneyear_df)
food17_pred_fouryear = regr_fouryear.predict(att_test_oneyear_df)
food17_pred_threeyear = regr_threeyear.predict(att_test_oneyear_df)
food17_pred_twoyear = regr_twoyear.predict(att_test_oneyear_df)
food17_pred_oneyear = regr_oneyear.predict(att_test_oneyear_df)


def halfweighted(preds, years):
    return (preds * (1/ 2**years))
    
 #####clean this with a function
food17_bagged_allyears = (
        food17_pred_17year * 0.000000762939453125 +
        food17_pred_16year * 0.0000152587890625 +
        food17_pred_15year * 0.000030517578125 +
        food17_pred_14year * 0.00006103515625 +
        food17_pred_13year * 0.000244140625 +
        food17_pred_12year * 0.000244140625 + 
        food17_pred_11year * 0.00048828125 +
        food17_pred_10year * 0.0009765625 +
        food17_pred_9year * 0.001953125 +
        food17_pred_8year * 0.00390625 +
        food17_pred_7year * 0.015625 +
        food17_pred_6year * 0.015625 +
        food17_pred_fiveyear * 0.03125 +
        food17_pred_fouryear * 0.0625 +
        food17_pred_threeyear * 0.125 +
        food17_pred_twoyear * 0.25 +
        food17_pred_oneyear * 0.5
        )
'''
food17_bagged_allyears = (
        halfweighted(food17_pred_17year, 17) +
        halfweighted(food17_pred_16year, 16) +
        halfweighted(food17_pred_15year, 15) +
        halfweighted(food17_pred_14year, 14) +
        halfweighted(food17_pred_13year, 13) +
        halfweighted(food17_pred_12year, 12) +
        halfweighted(food17_pred_11year, 11) +
        halfweighted(food17_pred_10year, 10) +
        halfweighted(food17_pred_9year, 9) +
        halfweighted(food17_pred_8year, 8) +
        halfweighted(food17_pred_7year, 7) +
        halfweighted(food17_pred_6year, 6) +
        halfweighted(food17_pred_fiveyear, 5) +
        halfweighted(food17_pred_threeyear, 3) +
        halfweighted(food17_pred_twoyear, 2) +
        halfweighted(food17_pred_oneyear, 1) +
        )
'''


    
print("weighted bootstraped aggregated 17 years for last year: ")    
# The mean squared error

print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_oneyear_df, food17_bagged_allyears))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_oneyear_df, food17_bagged_allyears))

print(food17_bagged_allyears)
print(food17_test_oneyear_df)


#BAGGGING STARTS HERE: just using last 10 years
    
food17_pred_17year = regr_17year.predict(att_test_oneyear_df)
food17_pred_16year = regr_16year.predict(att_test_oneyear_df)
food17_pred_15year = regr_15year.predict(att_test_oneyear_df)
food17_pred_14year = regr_14year.predict(att_test_oneyear_df)
food17_pred_13year = regr_13year.predict(att_test_oneyear_df)
food17_pred_12year = regr_12year.predict(att_test_oneyear_df)
food17_pred_11year = regr_11year.predict(att_test_oneyear_df)
food17_pred_10year = regr_10year.predict(att_test_oneyear_df)
food17_pred_9year = regr_9year.predict(att_test_oneyear_df)
food17_pred_8year = regr_8year.predict(att_test_oneyear_df)
food17_pred_7year = regr_7year.predict(att_test_oneyear_df)
food17_pred_6year = regr_6year.predict(att_test_oneyear_df)
food17_pred_fiveyear = regr_fiveyear.predict(att_test_oneyear_df)
food17_pred_fouryear = regr_fouryear.predict(att_test_oneyear_df)
food17_pred_threeyear = regr_threeyear.predict(att_test_oneyear_df)
food17_pred_twoyear = regr_twoyear.predict(att_test_oneyear_df)
food17_pred_oneyear = regr_oneyear.predict(att_test_oneyear_df)

food17_bagged_allyears = np.mean([

        food17_pred_10year,
        food17_pred_9year,
        food17_pred_8year,
        food17_pred_7year,
        food17_pred_6year,
        food17_pred_fiveyear,
        food17_pred_fouryear,
        food17_pred_threeyear,
        food17_pred_twoyear,
        food17_pred_oneyear
        ], axis = 0)
    

    
print("bootstraped aggregated 10 years for last year: ")    
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_oneyear_df, food17_bagged_allyears))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_oneyear_df, food17_bagged_allyears))


####################################################

#BAGGGING STARTS HERE: testing on 2015?
    
food17_pred_17year = regr_17year.predict(att_test_twoyear_df)
food17_pred_16year = regr_16year.predict(att_test_twoyear_df)
food17_pred_15year = regr_15year.predict(att_test_twoyear_df)
food17_pred_14year = regr_14year.predict(att_test_twoyear_df)
food17_pred_13year = regr_13year.predict(att_test_twoyear_df)
food17_pred_12year = regr_12year.predict(att_test_twoyear_df)
food17_pred_11year = regr_11year.predict(att_test_twoyear_df)
food17_pred_10year = regr_10year.predict(att_test_twoyear_df)
food17_pred_9year = regr_9year.predict(att_test_twoyear_df)
food17_pred_8year = regr_8year.predict(att_test_twoyear_df)
food17_pred_7year = regr_7year.predict(att_test_twoyear_df)
food17_pred_6year = regr_6year.predict(att_test_twoyear_df)
food17_pred_fiveyear = regr_fiveyear.predict(att_test_twoyear_df)
food17_pred_fouryear = regr_fouryear.predict(att_test_twoyear_df)
food17_pred_threeyear = regr_threeyear.predict(att_test_twoyear_df)
food17_pred_twoyear = regr_twoyear.predict(att_test_twoyear_df)
food17_pred_oneyear = regr_oneyear.predict(att_test_twoyear_df)

food17_bagged_allyears = np.mean([
        food17_pred_17year,
        food17_pred_16year,
        food17_pred_15year,
        food17_pred_14year,
        food17_pred_13year,
        food17_pred_12year,
        food17_pred_11year,
        food17_pred_10year,
        food17_pred_9year,
        food17_pred_8year,
        food17_pred_7year,
        food17_pred_6year,
        food17_pred_fiveyear,
        food17_pred_fouryear,
        food17_pred_threeyear,
        food17_pred_twoyear,
        food17_pred_oneyear
        ], axis = 0)
    
print(food17_bagged_allyears)
    
print("bootstraped aggregated 17 years for two year: ")    
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_oneyear_df, food17_bagged_allyears))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_twoyear_df, food17_bagged_allyears))

##################INFLATION ASJUSTED#################################### 1% inflation adjusted before averaging
    
food17_pred_17year = regr_17year.predict(att_test_oneyear_df)
food17_pred_16year = regr_16year.predict(att_test_oneyear_df)
food17_pred_15year = regr_15year.predict(att_test_oneyear_df)
food17_pred_14year = regr_14year.predict(att_test_oneyear_df)
food17_pred_13year = regr_13year.predict(att_test_oneyear_df)
food17_pred_12year = regr_12year.predict(att_test_oneyear_df)
food17_pred_11year = regr_11year.predict(att_test_oneyear_df)
food17_pred_10year = regr_10year.predict(att_test_oneyear_df)
food17_pred_9year = regr_9year.predict(att_test_oneyear_df)
food17_pred_8year = regr_8year.predict(att_test_oneyear_df)
food17_pred_7year = regr_7year.predict(att_test_oneyear_df)
food17_pred_6year = regr_6year.predict(att_test_oneyear_df)
food17_pred_fiveyear = regr_fiveyear.predict(att_test_oneyear_df)
food17_pred_fouryear = regr_fouryear.predict(att_test_oneyear_df)
food17_pred_threeyear = regr_threeyear.predict(att_test_oneyear_df)
food17_pred_twoyear = regr_twoyear.predict(att_test_oneyear_df)
food17_pred_oneyear = regr_oneyear.predict(att_test_oneyear_df)


food17_bagged_allyears = (
        food17_pred_17year * (1.01**16) +
        food17_pred_16year  * (1.01**15) +
        food17_pred_15year  * (1.01**14) +
        food17_pred_14year  * (1.01**13) +
        food17_pred_13year * (1.01**12) +
        food17_pred_12year * (1.01**11) +
        food17_pred_11year * (1.01**10) +
        food17_pred_10year * (1.01**9) +
        food17_pred_9year * (1.01**8) +
        food17_pred_8year * (1.01**7) +
        food17_pred_7year * (1.01**6) +
        food17_pred_6year * (1.01**5) +
        food17_pred_fiveyear * (1.01**4) +
        food17_pred_fouryear * (1.01**3) +
        food17_pred_threeyear * (1.01**2) +
        food17_pred_twoyear * (1.01**1) +
        food17_pred_oneyear 
        ) 

food17_bagged_allyears = np.divide(food17_bagged_allyears, 17)

print(food17_test_oneyear_df)
print(food17_bagged_allyears)
print(food17_pred_17year)


    

    
print("bootstraped aggregated iflation adjusted 17 years for last year: ")    
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_oneyear_df, food17_bagged_allyears))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_oneyear_df, food17_bagged_allyears))

######################################### inflation adjusted just 10 years 1%


    
food17_pred_17year = regr_17year.predict(att_test_oneyear_df)
food17_pred_16year = regr_16year.predict(att_test_oneyear_df)
food17_pred_15year = regr_15year.predict(att_test_oneyear_df)
food17_pred_14year = regr_14year.predict(att_test_oneyear_df)
food17_pred_13year = regr_13year.predict(att_test_oneyear_df)
food17_pred_12year = regr_12year.predict(att_test_oneyear_df)
food17_pred_11year = regr_11year.predict(att_test_oneyear_df)
food17_pred_10year = regr_10year.predict(att_test_oneyear_df)
food17_pred_9year = regr_9year.predict(att_test_oneyear_df)
food17_pred_8year = regr_8year.predict(att_test_oneyear_df)
food17_pred_7year = regr_7year.predict(att_test_oneyear_df)
food17_pred_6year = regr_6year.predict(att_test_oneyear_df)
food17_pred_fiveyear = regr_fiveyear.predict(att_test_oneyear_df)
food17_pred_fouryear = regr_fouryear.predict(att_test_oneyear_df)
food17_pred_threeyear = regr_threeyear.predict(att_test_oneyear_df)
food17_pred_twoyear = regr_twoyear.predict(att_test_oneyear_df)
food17_pred_oneyear = regr_oneyear.predict(att_test_oneyear_df)


food17_bagged_allyears = (
        food17_pred_10year * (1.01**9) +
        food17_pred_9year * (1.01**8) +
        food17_pred_8year * (1.01**7) +
        food17_pred_7year * (1.01**6) +
        food17_pred_6year * (1.01**5) +
        food17_pred_fiveyear * (1.01**4) +
        food17_pred_fouryear * (1.01**3) +
        food17_pred_threeyear * (1.01**2) +
        food17_pred_twoyear * (1.01**1) +
        food17_pred_oneyear 
        ) 

food17_bagged_allyears = np.divide(food17_bagged_allyears, 10)

print(food17_bagged_allyears)

print(food17_pred_10year *(1.01**9))

    

    
print("bootstraped aggregated iflation adjusted 10 years for last year: ")    
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_oneyear_df, food17_bagged_allyears))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_oneyear_df, food17_bagged_allyears))

##################################i dunno############################################

food17_pred_17year = regr_17year.predict(att_test_oneyear_df)
food17_pred_16year = regr_16year.predict(att_test_oneyear_df)
food17_pred_15year = regr_15year.predict(att_test_oneyear_df)
food17_pred_14year = regr_14year.predict(att_test_oneyear_df)
food17_pred_13year = regr_13year.predict(att_test_oneyear_df)
food17_pred_12year = regr_12year.predict(att_test_oneyear_df)
food17_pred_11year = regr_11year.predict(att_test_oneyear_df)
food17_pred_10year = regr_10year.predict(att_test_oneyear_df)
food17_pred_9year = regr_9year.predict(att_test_oneyear_df)
food17_pred_8year = regr_8year.predict(att_test_oneyear_df)
food17_pred_7year = regr_7year.predict(att_test_oneyear_df)
food17_pred_6year = regr_6year.predict(att_test_oneyear_df)
food17_pred_fiveyear = regr_fiveyear.predict(att_test_oneyear_df)
food17_pred_fouryear = regr_fouryear.predict(att_test_oneyear_df)
food17_pred_threeyear = regr_threeyear.predict(att_test_oneyear_df)
food17_pred_twoyear = regr_twoyear.predict(att_test_oneyear_df)
food17_pred_oneyear = regr_oneyear.predict(att_test_oneyear_df)

food17_bagged_allyears = np.mean([

        food17_pred_10year,
        food17_pred_9year,
        food17_pred_8year,
        food17_pred_7year,
        food17_pred_6year,
        food17_pred_fiveyear,
        food17_pred_fouryear,
        food17_pred_threeyear,
        food17_pred_twoyear,
        food17_pred_oneyear
        ], axis = 0)
    
    
print("bootstraped aggregated 17 years for last year ( four lowest removed): ")    
# The mean squared error
print("Mean squared error: %.2f"
      % mean_absolute_error(food17_test_oneyear_df, food17_bagged_allyears))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_oneyear_df, food17_bagged_allyears))
