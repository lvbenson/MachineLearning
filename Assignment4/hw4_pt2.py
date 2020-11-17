import numpy as np 
import pandas as pd 
import graphviz
from graphviz import render
from graphviz import Digraph
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
#build a decision tree for classifying whether a person has a college degree by greedily choosing threshold splits that maximize information gain. 

def decision_tree(data, column, value):
    
    left = data[data[column] <= value].index #meets the condition
    right = data[data[column] > value].index #do not meet the condition
    #print(left, right)
    return left, right




data = pd.DataFrame({
        "age": [24,53,23,25,32,52,22,43,52,48],
        "salary": [40000,52000,25000,77000,48000,110000,38000,44000,27000,65000],
        "degree": ['yes','no','no','yes','yes','yes','yes','no','no','yes'],
    })

X = data.iloc[:,:-1] # Features 
y = data.iloc[:,-1] # Target 
l, r = decision_tree(X, "salary",27000)


def gini_impurity(label, label_idx):
    
    unique_label, unique_label_count = np.unique(label.loc[label_idx], return_counts=True)

    impurity = 1.0
    for i in range(len(unique_label)):
        p_i = unique_label_count[i] / sum(unique_label_count)
        impurity -= p_i ** 2 
    return impurity

g = gini_impurity(y,y.index)

def information_gain(label, left_idx, right_idx, impurity): 

    p = float(len(left_idx)) / (len(left_idx) + len(right_idx))
    info_gain = impurity - p * gini_impurity(label, left_idx) - (1 - p) * gini_impurity(label, right_idx)
    #print(info_gain)
    return info_gain

i_g = information_gain(y,l,r,g)

def find_best_split(df, label, idx):

    best_gain = 0 
    best_col = None
    best_value = None
    
    df = df.loc[idx] # converting training data to pandas dataframe
    label_idx = label.loc[idx].index # getting the index of the labels

    impurity = gini_impurity(label, label_idx) # determining the impurity at the current node
    
    # go through the columns and store the unique values in each column (no point testing on the same value twice)
    for col in df.columns: 
        unique_values = set(df[col])
        # loop thorugh each value and partition the data into true (left_index) and false (right_index)
        for value in unique_values: 

            left_idx, right_idx = decision_tree(df, col, value)
            # ignore if the index is empty (meaning there was no features that met the decision rule)
            if len(left_idx) == 0 or len(right_idx) == 0: 
                continue 
            # determine the info gain at the node
            info_gain = information_gain(label, left_idx, right_idx, impurity)
            # if the info gain is higher then our current best gain then that becomes the best gain
            if info_gain > best_gain:
                best_gain, best_col, best_value = info_gain, col, value
                
    return best_gain, best_col, best_value

gain,col,val = find_best_split(X, y, y.index)
print(gain,col,val)

# helper function to count values
def count(label, idx):

    unique_label, unique_label_counts = np.unique(label.loc[idx], return_counts=True)
    dict_label_count = dict(zip(unique_label, unique_label_counts))
    return dict_label_count
# check counts at first node to check it aligns with sci-kit learn
d = count(y, y.index)
print(d)