import os
import numpy as np
import pandas as pd
import sys
import name2vector

#Problem 1

#x is the names column
boys = pd.read_csv('/Users/lvbenson/MachineLearning_Fall2020/Assignment2/boy_names.csv')
#print(boys)
girls = pd.read_csv('/Users/lvbenson/MachineLearning_Fall2020/Assignment2/girl_names.csv')
#print(girls)
#look at names

#First, for the test set
#create dataframe of names and corresponding gender. Label gender as male and female.

names = pd.DataFrame({'name':list(boys['x'])+list(girls['x']),
'gender':['m' for b in range(len(boys['x']))]+['f' for g in range(len(girls['x']))]})

#creates a new dataframe of names, gender, and features (prefix, suffix) 
feat = name2vector.features(names)
name2vector.train(names)

test = pd.read_csv('/Users/lvbenson/MachineLearning_Fall2020/Assignment2/test_names.csv')

test_names = pd.DataFrame({'name':list(test['x'])})
#print(test_names)
name2vector.test(names, test_names)