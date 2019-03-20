# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 20:37:12 2018

@author: Patrick
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_regression, mutual_info_regression


# data set from jay divided into two files, and august 2017 removed
attributes = "2018_Attributes.csv"
attributes_df = pd.read_csv(attributes)
#targets = "2018_Targets_2017.csv"
targets = "2018_Targets.csv"
#targets dataframe contains all groups including Food 17
targets_df = pd.read_csv(targets)

# 170 is January 1st 1999
# -2 to take of jan and feb of 2018
#targets_df = targets_df[170:-2]

targets_df = targets_df[170:-7]
# -7 to take off months of 2017
attributes_df = attributes_df[170:-7]

#individual target dataframes
CPI_df = targets_df['Food 17']
restaurants_df = targets_df['restaurants 17']
vegetables_df = targets_df['Vegetables ']
other_df = targets_df['Other ']
meat_df = targets_df['Meat']
dairyandeggs_df = targets_df['Dairy products and eggs']
bakery_df = targets_df['Bakery']
fruit_df = targets_df['Fruit']
allitems_df = targets_df['All-items']
fromstores_df = targets_df['Food purchased from stores']
beef_df = targets_df['Fresh or frozen beef']
pork_df = targets_df['Fresh or frozen pork']
chicken_df = targets_df['Fresh or frozen chicken']
dairy_df = targets_df['Dairy products']
eggs_df = targets_df['Eggs']
coffee_df = targets_df['Coffee']
babyfood_df = targets_df['Baby foods']
shelter18_df = targets_df['Shelter 18']
transportation_df = targets_df['Transportation']
gas_df = targets_df['Gasoline']
energy25_df = targets_df['Energy 25']
fishandseafood_df = targets_df['Fish seafood']

#switches 
restaurants = True
vegetables = True
other = True
meat = True
dairyandeggs = True
bakery = True
fruit = True
allitems = True
fromstores = True
beef = True
pork = True
chicken = True
dairy = True
eggs = True
coffee = True
babyfood = True
shelter = True
transportation = True
gas = True
energy = True
fishandseafood = True

print("Attributes DataFrame:")
print(str(attributes_df.info()))

print("Targets DataFrame:")
print(str(targets_df.info()))

print("Food 17 Dataframe:")
#print(str(CPI_df.info()))

timestamps_df = targets_df['Timestamp']

food17_target_df =  CPI_df

# remoe the timestamps from both
targets_df = targets_df.drop('Timestamp', axis = 1)
attributes_df = attributes_df.drop('Timestamp', axis = 1)

# drop all attributes that are missing values for now
# revise this later when we predict them.
attributes_df =attributes_df.dropna(how='any', axis = 1)

print(attributes_df.info())
att_df= attributes_df

att_df = attributes_df
#for x in range(5,30):
if True:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=29 )
    attributes_df = selector.fit_transform(attributes_df,food17_target_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("food 17:")
    print(features_dataframe_new.info())
    
    oneyear = -12
    #print(attributes_df)
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12]
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    food17_train_oneyear_df =food17_target_df[:oneyear]
    food17_test_oneyear_df =food17_target_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, food17_train_oneyear_df)
    # Make predictions using the testing set
    food17_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(food17_test_oneyear_df, food17_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(food17_test_oneyear_df, food17_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,food17_test_oneyear_df,  color='black')
    plt.plot(lastyear,food17_pred_oneyear, color='blue', linewidth=3)
    #plt.axis([lastyear[0]-3,lastyear[11]+3,135,148])
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    #need to write script to compute dates from timestamps!
    plt.title('2017-01-01 to 2018-01-01')
    plt.show()

################ restaurants ##################################### 
if restaurants:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=29 )
    attributes_df = selector.fit_transform(attributes_df,restaurants_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("restaurants:")
    print(features_dataframe_new.info())
    oneyear = -12
    #print(attributes_df)
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12]
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    restaurants_train_oneyear_df =restaurants_df[:oneyear]
    restaurants_test_oneyear_df =restaurants_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, restaurants_train_oneyear_df)
    # Make predictions using the testing set
    restaurants_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(restaurants_test_oneyear_df, restaurants_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(restaurants_test_oneyear_df, restaurants_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,restaurants_test_oneyear_df,  color='black')
    plt.plot(lastyear,restaurants_pred_oneyear, color='blue', linewidth=3)
    #plt.axis([lastyear[0]-3,lastyear[11]+3,135,148])
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    #need to write script to compute dates from timestamps!
    plt.title(' Restraurants 2017-01-01 to 2018-01-01')
    plt.show()

################ vegetables ##################################### 
if vegetables:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=15 )
    attributes_df = selector.fit_transform(attributes_df,vegetables_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("vegetables")
    print(features_dataframe_new.info())
    
    oneyear = -12
    #print(attributes_df)
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    vegetables_train_oneyear_df =vegetables_df[:oneyear]
    vegetables_test_oneyear_df =vegetables_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, vegetables_train_oneyear_df)
    # Make predictions using the testing set
    vegetables_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(vegetables_test_oneyear_df, vegetables_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(vegetables_test_oneyear_df, vegetables_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,vegetables_test_oneyear_df,  color='black')
    plt.plot(lastyear,vegetables_pred_oneyear, color='blue', linewidth=3)
    #plt.axis([lastyear[0]-3,lastyear[11]+3,135,148])
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    #need to write script to compute dates from timestamps!
    plt.title('vegetables 2017-01-01 to 2018-01-01')
    plt.show()

################ other ##################################### 
#for x in range(5,30):
if other:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=10 )
    attributes_df = selector.fit_transform(attributes_df,other_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print(features_dataframe_new.info())
    print("other")    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    other_train_oneyear_df =other_df[:oneyear]
    other_test_oneyear_df =other_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, other_train_oneyear_df)
    # Make predictions using the testing set
    other_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(other_test_oneyear_df, other_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(other_test_oneyear_df, other_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,other_test_oneyear_df,  color='black')
    plt.plot(lastyear,other_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('other 2017-01-01 to 2018-01-01')
    plt.show()

################ meat ##################################### 
#for x in range(5,30):
if meat:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=29 )
    attributes_df = selector.fit_transform(attributes_df,meat_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("meat:")
    print(features_dataframe_new.info())
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    meat_train_oneyear_df =meat_df[:oneyear]
    meat_test_oneyear_df =meat_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, meat_train_oneyear_df)
    # Make predictions using the testing set
    meat_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(meat_test_oneyear_df, meat_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(meat_test_oneyear_df, meat_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,meat_test_oneyear_df,  color='black')
    plt.plot(lastyear,meat_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('meat 2017-01-01 to 2018-01-01')
    plt.show()

################ dairyandeggs ##################################### 
#for x in range(5,30):
if dairyandeggs:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=29 )
    attributes_df = selector.fit_transform(attributes_df,dairyandeggs_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("dairy and eggs:")
    print(features_dataframe_new.info())
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    dairyandeggs_train_oneyear_df =dairyandeggs_df[:oneyear]
    dairyandeggs_test_oneyear_df =dairyandeggs_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, dairyandeggs_train_oneyear_df)
    # Make predictions using the testing set
    dairyandeggs_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(dairyandeggs_test_oneyear_df, dairyandeggs_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(dairyandeggs_test_oneyear_df, dairyandeggs_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,dairyandeggs_test_oneyear_df,  color='black')
    plt.plot(lastyear,dairyandeggs_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('dairyandeggs 2017-01-01 to 2018-01-01')
    plt.show()

################ bakery ##################################### 
#for x in range(5,30):
if bakery:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=29 )
    attributes_df = selector.fit_transform(attributes_df,bakery_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("bakery:")  
    print(features_dataframe_new.info())
 
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    bakery_train_oneyear_df =bakery_df[:oneyear]
    bakery_test_oneyear_df =bakery_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, bakery_train_oneyear_df)
    # Make predictions using the testing set
    bakery_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(bakery_test_oneyear_df, bakery_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(bakery_test_oneyear_df, bakery_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,bakery_test_oneyear_df,  color='black')
    plt.plot(lastyear,bakery_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('bakery 2017-01-01 to 2018-01-01')
    plt.show()

################ fruit ##################################### 
#for x in range(5,30):
if fruit:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=29 )
    attributes_df = selector.fit_transform(attributes_df,fruit_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("fruit")     
    print(features_dataframe_new.info())

    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    fruit_train_oneyear_df =fruit_df[:oneyear]
    fruit_test_oneyear_df =fruit_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, fruit_train_oneyear_df)
    # Make predictions using the testing set
    fruit_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(fruit_test_oneyear_df, fruit_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(fruit_test_oneyear_df, fruit_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,fruit_test_oneyear_df,  color='black')
    plt.plot(lastyear,fruit_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('fruit 2017-01-01 to 2018-01-01')
    plt.show()

################ allitems ##################################### 
#for x in range(5,30):
if allitems:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=18 )
    attributes_df = selector.fit_transform(attributes_df,allitems_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("all items:")
    print(features_dataframe_new.info())

    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    allitems_train_oneyear_df =allitems_df[:oneyear]
    allitems_test_oneyear_df =allitems_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, allitems_train_oneyear_df)
    # Make predictions using the testing set
    allitems_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(allitems_test_oneyear_df, allitems_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(allitems_test_oneyear_df, allitems_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,allitems_test_oneyear_df,  color='black')
    plt.plot(lastyear,allitems_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('allitems 2017-01-01 to 2018-01-01')
    plt.show()

################ fromstores ##################################### 
#for x in range(5,30):
if fromstores:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=29 )
    attributes_df = selector.fit_transform(attributes_df,fromstores_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("fromstores:")
    print(features_dataframe_new.info())
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    fromstores_train_oneyear_df =fromstores_df[:oneyear]
    fromstores_test_oneyear_df =fromstores_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, fromstores_train_oneyear_df)
    # Make predictions using the testing set
    fromstores_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(fromstores_test_oneyear_df, fromstores_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(fromstores_test_oneyear_df, fromstores_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,fromstores_test_oneyear_df,  color='black')
    plt.plot(lastyear,fromstores_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('fromstores 2017-01-01 to 2018-01-01')
    plt.show()

################ beef ##################################### 
#for x in range(5,30):
if beef:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=20 )
    attributes_df = selector.fit_transform(attributes_df,beef_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("beef:")
    print(features_dataframe_new.info())
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    beef_train_oneyear_df =beef_df[:oneyear]
    beef_test_oneyear_df =beef_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, beef_train_oneyear_df)
    # Make predictions using the testing set
    beef_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(beef_test_oneyear_df, beef_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(beef_test_oneyear_df, beef_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,beef_test_oneyear_df,  color='black')
    plt.plot(lastyear,beef_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('beef 2017-01-01 to 2018-01-01')
    plt.show()

################ pork ##################################### 
#for x in range(5,30):
if pork:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=29 )
    attributes_df = selector.fit_transform(attributes_df,pork_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("pork:")
    print(features_dataframe_new.info())
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    pork_train_oneyear_df =pork_df[:oneyear]
    pork_test_oneyear_df =pork_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, pork_train_oneyear_df)
    # Make predictions using the testing set
    pork_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(pork_test_oneyear_df, pork_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(pork_test_oneyear_df, pork_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,pork_test_oneyear_df,  color='black')
    plt.plot(lastyear,pork_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('pork 2017-01-01 to 2018-01-01')
    plt.show()

################ chicken ##################################### 
#for x in range(5,30):
if chicken:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=29 )
    attributes_df = selector.fit_transform(attributes_df,chicken_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("chicken:")
    print(features_dataframe_new.info())
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    chicken_train_oneyear_df =chicken_df[:oneyear]
    chicken_test_oneyear_df =chicken_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, chicken_train_oneyear_df)
    # Make predictions using the testing set
    chicken_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(chicken_test_oneyear_df, chicken_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(chicken_test_oneyear_df, chicken_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,chicken_test_oneyear_df,  color='black')
    plt.plot(lastyear,chicken_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('chicken 2016-01-01 to 2017-01-01')
    plt.show()

################ dairy ##################################### 
#for x in range(5,30):
if dairy:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=29 )
    attributes_df = selector.fit_transform(attributes_df,dairy_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("dairy:")
    print(features_dataframe_new.info())
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    dairy_train_oneyear_df =dairy_df[:oneyear]
    dairy_test_oneyear_df =dairy_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, dairy_train_oneyear_df)
    # Make predictions using the testing set
    dairy_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(dairy_test_oneyear_df, dairy_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(dairy_test_oneyear_df, dairy_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,dairy_test_oneyear_df,  color='black')
    plt.plot(lastyear,dairy_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('dairy 2017-01-01 to 2018-01-01')
    plt.show()

################ eggs ##################################### 
#for x in range(5,30):
if eggs:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=29 )
    attributes_df = selector.fit_transform(attributes_df,eggs_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("eggs:")
    print(features_dataframe_new.info())
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    eggs_train_oneyear_df =eggs_df[:oneyear]
    eggs_test_oneyear_df =eggs_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, eggs_train_oneyear_df)
    # Make predictions using the testing set
    eggs_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(eggs_test_oneyear_df, eggs_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(eggs_test_oneyear_df, eggs_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,eggs_test_oneyear_df,  color='black')
    plt.plot(lastyear,eggs_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('eggs 2017-01-01 to 2018-01-01')
    plt.show()

################ coffee ##################################### 
#for x in range(5,30):
if coffee:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=28 )
    attributes_df = selector.fit_transform(attributes_df,coffee_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("coffee:")
    print(features_dataframe_new.info())
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    coffee_train_oneyear_df =coffee_df[:oneyear]
    coffee_test_oneyear_df =coffee_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, coffee_train_oneyear_df)
    # Make predictions using the testing set
    coffee_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(coffee_test_oneyear_df, coffee_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(coffee_test_oneyear_df, coffee_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
        
    # Plot outputs
    plt.scatter(lastyear,coffee_test_oneyear_df,  color='black')
    plt.plot(lastyear,coffee_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('coffee 2016-01-01 to 2017-01-01')
    plt.show()

################ babyfood ##################################### 
#for x in range(5,30):
if babyfood:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=12 )
    attributes_df = selector.fit_transform(attributes_df,babyfood_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("babyfood:")
    print(features_dataframe_new.info())
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    babyfood_train_oneyear_df =babyfood_df[:oneyear]
    babyfood_test_oneyear_df =babyfood_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, babyfood_train_oneyear_df)
    # Make predictions using the testing set
    babyfood_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(babyfood_test_oneyear_df, babyfood_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(babyfood_test_oneyear_df, babyfood_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,babyfood_test_oneyear_df,  color='black')
    plt.plot(lastyear,babyfood_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('babyfood 2017-01-01 to 2018-01-01')
    plt.show()

################ shelter18 ##################################### 
#for x in range(5,30):
if shelter:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=29 )
    attributes_df = selector.fit_transform(attributes_df,shelter18_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("shelter18:")
    print(features_dataframe_new.info())
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    shelter18_train_oneyear_df =shelter18_df[:oneyear]
    shelter18_test_oneyear_df =shelter18_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, shelter18_train_oneyear_df)
    # Make predictions using the testing set
    shelter18_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(shelter18_test_oneyear_df, shelter18_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(shelter18_test_oneyear_df, shelter18_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,shelter18_test_oneyear_df,  color='black')
    plt.plot(lastyear,shelter18_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('shelter18 2017-01-01 to 2018-01-01')
    plt.show()

################ transportation ##################################### 
#for x in range(5,30):
if transportation:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=20 )
    attributes_df = selector.fit_transform(attributes_df,transportation_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("transportation:")
    print(features_dataframe_new.info())
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    transportation_train_oneyear_df =transportation_df[:oneyear]
    transportation_test_oneyear_df =transportation_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, transportation_train_oneyear_df)
    # Make predictions using the testing set
    transportation_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(transportation_test_oneyear_df, transportation_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(transportation_test_oneyear_df, transportation_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,transportation_test_oneyear_df,  color='black')
    plt.plot(lastyear,transportation_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('transportation 2017-01-01 to 2018-01-01')
    plt.show()

################ gas ##################################### 
#for x in range(5,30):
if gas:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=20 )
    attributes_df = selector.fit_transform(attributes_df,gas_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("gas:")
    print(features_dataframe_new.info())
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    gas_train_oneyear_df =gas_df[:oneyear]
    gas_test_oneyear_df =gas_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, gas_train_oneyear_df)
    # Make predictions using the testing set
    gas_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(gas_test_oneyear_df, gas_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(gas_test_oneyear_df, gas_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,gas_test_oneyear_df,  color='black')
    plt.plot(lastyear,gas_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('gas 2017-01-01 to 2018-01-01')
    plt.show()

################ energy25 ##################################### 
#for x in range(5,30):
if energy:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=20 )
    attributes_df = selector.fit_transform(attributes_df,energy25_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("energy25:")
    print(features_dataframe_new.info())
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    energy25_train_oneyear_df =energy25_df[:oneyear]
    energy25_test_oneyear_df =energy25_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, energy25_train_oneyear_df)
    # Make predictions using the testing set
    energy25_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(energy25_test_oneyear_df, energy25_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(energy25_test_oneyear_df, energy25_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,energy25_test_oneyear_df,  color='black')
    plt.plot(lastyear,energy25_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('energy25 2017-01-01 to 2018-01-01')
    plt.show()

################ fishandseafood ##################################### 
#for x in range(5,30):
if fishandseafood:
    attributes_df=att_df
    selector = SelectKBest(mutual_info_regression,k=13 )
    attributes_df = selector.fit_transform(attributes_df,fishandseafood_df)
    
    idxs_selected = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    features_dataframe_new = att_df[idxs_selected]
    print("fishandseafood:")
    print(features_dataframe_new.info())
    
    #2017 data added to targets:
    att_train_oneyear_df = attributes_df[:-12] 
    #target groupings
    tar_train_oneyear__df = targets_df[:oneyear]
    tar_test_oneyear_df = targets_df[oneyear:]
    # food 17 alone
    fishandseafood_train_oneyear_df =fishandseafood_df[:oneyear]
    fishandseafood_test_oneyear_df =fishandseafood_df[oneyear:]
    
    # Create linear regression object for one year
    regr_oneyear = linear_model.LinearRegression()
    # Train the model using the training sets
    regr_oneyear.fit(att_train_oneyear_df, fishandseafood_train_oneyear_df)
    # Make predictions using the testing set
    fishandseafood_pred_oneyear = regr_oneyear.predict(att_train_oneyear_df[-12:])
    
    # The mean absolute error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(fishandseafood_test_oneyear_df, fishandseafood_pred_oneyear))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(fishandseafood_test_oneyear_df, fishandseafood_pred_oneyear))
    
    #target_CPI_pred.plot()
    lastyear = timestamps_df[oneyear:]
    lastyear= lastyear.as_matrix()
    
    # Plot outputs
    plt.scatter(lastyear,fishandseafood_test_oneyear_df,  color='black')
    plt.plot(lastyear,fishandseafood_pred_oneyear, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    plt.ylabel('$')
    plt.xlabel('Months')
    plt.title('fishandseafood 2017-01-01 to 2018-01-01')
    plt.show()