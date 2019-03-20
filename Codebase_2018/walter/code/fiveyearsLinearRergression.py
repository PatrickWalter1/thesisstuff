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

#cant get this to work right (trying to remove hist before 1999)
#targets_df = targets_df[int(targets_df.Timestamp) >= 36161]
#attributes_df = attributes_df[int(attributes_df.Timestamp) >= 36161]

CPI_df = targets_df['Food 17']

print("Attributes DataFrame:")
print(str(attributes_df.info()))

print("Targets DataFrame:")
print(str(targets_df.info()))

print("Food 17 Dataframe:")
#print(str(CPI_df.info()))

timestamps_df = attributes_df['Timestamp']
food17_target_df =  CPI_df

# remoe the timestamps from both
targets_df = targets_df.drop('Timestamp', axis = 1)
attributes_df = attributes_df.drop('Timestamp', axis = 1)

# drop all attributes that are missing values for now
# revise this later when we predict them.
attributes_df =attributes_df.dropna(how='any', axis = 1)

print("New taregsts dataframe:")
print(attributes_df.info())
#print(attributes_df.head())
#print(attributes_df.head())
#attributes_df = SelectKBest(mutual_info_regression,k= 7).fit_transform(attributes_df,food17_target_df)
#how much to take off timeframe in months negative
#will be used for predicting each year based on years before
#currently doesnt line up with years since last record is July 2017
fiveyear = -60 #months
fouryear = -48
threeyear = -36
twoyear = -24
oneyear = -12
sixmonth = -6

#one year divide into testing and training 
#attributes
att_train_oneyear_df = attributes_df[:oneyear]
att_test_oneyear_df = attributes_df.iloc[oneyear:]
#target groupings
tar_train_oneyear__df = targets_df[:oneyear]
tar_test_oneyear_df = targets_df.iloc[oneyear:]
# food 17 alone
food17_train_oneyear_df =food17_target_df[:oneyear]
food17_test_oneyear_df =food17_target_df.iloc[oneyear:]

#last six months
#attributes
att_train_sixmonth_df = attributes_df[:sixmonth]
att_test_sixmonth_df = attributes_df.iloc[sixmonth:]
#target groupings
tar_train_sixmonth__df = targets_df[:sixmonth]
tar_test_sixmonth_df = targets_df.iloc[sixmonth:]
# food 17 alone
food17_train_sixmonth_df =food17_target_df[:sixmonth]
food17_test_sixmonth_df =food17_target_df.iloc[sixmonth:]

#2 years ago
#attributes
att_train_twoyear_df = attributes_df[:twoyear]
att_test_twoyear_df = attributes_df.iloc[twoyear:oneyear]
#target groupings
tar_train_twoyear__df = targets_df[:twoyear]
tar_test_twoyear_df = targets_df.iloc[twoyear:oneyear]
# food 17 alone
food17_train_twoyear_df =food17_target_df[:twoyear]
food17_test_twoyear_df =food17_target_df.iloc[twoyear:oneyear]

#3 years ago
#attributes
att_train_threeyear_df = attributes_df[:threeyear]
att_test_threeyear_df = attributes_df.iloc[threeyear:twoyear]
#target groupings
tar_train_threeyear__df = targets_df[:threeyear]
tar_test_threeyear_df = targets_df.iloc[threeyear:twoyear]
# food 17 alone
food17_train_threeyear_df =food17_target_df[:threeyear]
food17_test_threeyear_df =food17_target_df.iloc[threeyear:twoyear]

#4 years ago
#attributes
att_train_fouryear_df = attributes_df[:fouryear]
att_test_fouryear_df = attributes_df.iloc[fouryear:threeyear]
#target groupings
tar_train_fouryear__df = targets_df[:fouryear]
tar_test_fouryear_df = targets_df.iloc[fouryear:threeyear]
# food 17 alone
food17_train_fouryear_df =food17_target_df[:fouryear]
food17_test_fouryear_df =food17_target_df.iloc[fouryear:threeyear]

#5 years ago
#attributes
att_train_fiveyear_df = attributes_df[:fiveyear]
att_test_fiveyear_df = attributes_df.iloc[fiveyear:fouryear]
#target groupings
tar_train_fiveyear__df = targets_df[:fiveyear]
tar_test_fiveyear_df = targets_df.iloc[fiveyear:fouryear]
# food 17 alone
food17_train_fiveyear_df =food17_target_df[:fiveyear]
food17_test_fiveyear_df =food17_target_df.iloc[fiveyear:fouryear]

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
print("last 6 months back train and test shapes:")
print(att_train_sixmonth_df.shape[0], att_train_oneyear_df.shape[1])
print(att_test_sixmonth_df.shape[0], att_test_oneyear_df.shape[1])


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
print("2016-08-01 to 2017-07-01")
print('Coefficients: \n', regr_oneyear.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(food17_test_oneyear_df, food17_pred_oneyear))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_oneyear_df, food17_pred_oneyear))

#target_CPI_pred.plot()
lastyear = timestamps_df.iloc[oneyear:]
lastyear= lastyear.as_matrix()
#print(lastyear)
# Plot outputs
plt.scatter(lastyear,food17_test_oneyear_df,  color='black')
plt.plot(lastyear,food17_pred_oneyear, color='blue', linewidth=3)
#plt.axis([lastyear[0]-3,lastyear[11]+3,135,148])
plt.xticks(())
plt.yticks(())
plt.ylabel('Food 17')
plt.xlabel('Months')
#need to write script to compute dates from timestamps!
plt.title('2016-08-01 to 2017-07-01')
plt.show()

#TWO YEARS BACK 12 MONTHS!
regr_twoyear = linear_model.LinearRegression()
# Train the model using the training sets
regr_twoyear.fit(att_train_twoyear_df, food17_train_twoyear_df)
# Make predictions using the testing set
food17_pred_twoyear = regr_twoyear.predict(att_test_twoyear_df)
# The coefficients
print("2015-08-01 to 2016-07-01")
print('Coefficients: \n', regr_twoyear.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(food17_test_twoyear_df, food17_pred_twoyear))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_twoyear_df, food17_pred_twoyear))

#target_CPI_pred.plot()
lastyear = timestamps_df.iloc[twoyear:oneyear]
lastyear= lastyear.as_matrix()
#print(lastyear)
# Plot outputs
plt.scatter(lastyear,food17_test_twoyear_df,  color='black')
plt.plot(lastyear,food17_pred_twoyear, color='blue', linewidth=3)
#plt.axis([lastyear[0]-3,lastyear[11]+3,135,148])
plt.xticks(())
plt.yticks(())
plt.ylabel('Food 17')
plt.xlabel('Months')
#need to write script to compute dates from timestamps!
plt.title('2015-08-01 to 2016-07-01')
plt.show()

#THREE YEARS BACK 12 MONTHS!
regr_threeyear = linear_model.LinearRegression()
# Train the model using the training sets
regr_threeyear.fit(att_train_threeyear_df, food17_train_threeyear_df)
# Make predictions using the testing set
food17_pred_threeyear = regr_threeyear.predict(att_test_threeyear_df)
# The coefficients
print("2014-08-01 to 2015-07-01")
print('Coefficients: \n', regr_threeyear.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(food17_test_threeyear_df, food17_pred_threeyear))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_threeyear_df, food17_pred_threeyear))

#target_CPI_pred.plot()
lastyear = timestamps_df.iloc[threeyear:twoyear]
lastyear= lastyear.as_matrix()
#print(lastyear)
# Plot outputs
plt.scatter(lastyear,food17_test_threeyear_df,  color='black')
plt.plot(lastyear,food17_pred_threeyear, color='blue', linewidth=3)
#plt.axis([lastyear[0]-3,lastyear[11]+3,135,148])
plt.xticks(())
plt.yticks(())
plt.ylabel('Food 17')
plt.xlabel('Months')
#need to write script to compute dates from timestamps!
plt.title('2014-08-01 to 2015-07-01')
plt.show()

#FOUR YEARS BACK 12 MONTHS!
regr_fouryear = linear_model.LinearRegression()
# Train the model using the training sets
regr_fouryear.fit(att_train_fouryear_df, food17_train_fouryear_df)
# Make predictions using the testing set
food17_pred_fouryear = regr_fouryear.predict(att_test_fouryear_df)
# The coefficients
print("2013-08-01 to 2014-07-01")
print('Coefficients: \n', regr_fouryear.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(food17_test_fouryear_df, food17_pred_fouryear))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_fouryear_df, food17_pred_fouryear))

#target_CPI_pred.plot()
lastyear = timestamps_df.iloc[fouryear:threeyear]
lastyear= lastyear.as_matrix()
#print(lastyear)
# Plot outputs
plt.scatter(lastyear,food17_test_fouryear_df,  color='black')
plt.plot(lastyear,food17_pred_fouryear, color='blue', linewidth=3)
#plt.axis([lastyear[0]-3,lastyear[11]+3,135,148])
plt.xticks(())
plt.yticks(())
plt.ylabel('Food 17')
plt.xlabel('Months')
#need to write script to compute dates from timestamps!
plt.title('2013-08-01 to 2014-07-01')
plt.show()

#FIVE YEARS BACK 12 MONTHS!
regr_fiveyear = linear_model.LinearRegression()
# Train the model using the training sets
regr_fiveyear.fit(att_train_fiveyear_df, food17_train_fiveyear_df)
# Make predictions using the testing set
food17_pred_fiveyear = regr_fiveyear.predict(att_test_fiveyear_df)
# The coefficients
print("2012-08-01 to 2013-07-01")
print('Coefficients: \n', regr_fiveyear.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(food17_test_fiveyear_df, food17_pred_fiveyear))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(food17_test_fiveyear_df, food17_pred_fiveyear))

#target_CPI_pred.plot()
lastyear = timestamps_df.iloc[fiveyear:fouryear]
lastyear= lastyear.as_matrix()
#print(lastyear)
# Plot outputs
plt.scatter(lastyear,food17_test_fiveyear_df,  color='black')
plt.plot(lastyear,food17_pred_fiveyear, color='blue', linewidth=3)
#plt.axis([lastyear[0]-3,lastyear[11]+3,135,148])
plt.xticks(())
plt.yticks(())
plt.ylabel('Food 17')
plt.xlabel('Months')
#need to write script to compute dates from timestamps!
plt.title('2012-08-01 to 2013-07-01')
plt.show()


lastyear = timestamps_df.iloc[fiveyear:]
lastyear= lastyear.as_matrix()
#print(lastyear)
# Plot outputs
plt.scatter(lastyear[oneyear:],food17_test_oneyear_df,  color='black')
plt.plot(lastyear[oneyear:],food17_pred_oneyear, color='blue', linewidth=3)

plt.scatter(lastyear[twoyear:oneyear],food17_test_twoyear_df,  color='black')
plt.plot(lastyear[twoyear:oneyear],food17_pred_twoyear, color='red', linewidth=3)

plt.scatter(lastyear[threeyear:twoyear],food17_test_threeyear_df,  color='black')
plt.plot(lastyear[threeyear:twoyear],food17_pred_threeyear, color='yellow', linewidth=3)

plt.scatter(lastyear[fouryear:threeyear],food17_test_fouryear_df,  color='black')
plt.plot(lastyear[fouryear:threeyear],food17_pred_fouryear, color='green', linewidth=3)

plt.plot(lastyear,food17_target_df[fiveyear:])

plt.scatter(lastyear[:fouryear],food17_test_fiveyear_df,  color='black')
plt.plot(lastyear[:fouryear],food17_pred_fiveyear, color='blue', linewidth=3)



plt.xticks(())
plt.yticks(())
plt.ylabel('Food 17')
plt.xlabel('Months')
#need to write script to compute dates from timestamps!
plt.title('2012-08-01 to 2017-07-01')
plt.show()