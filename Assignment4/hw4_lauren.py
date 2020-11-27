#refer:https://github.com/danielpang/decision-trees/blob/master/learn.py
import pandas as pd
import numpy as np

dataq2 = pd.DataFrame.from_dict({'age': pd.Series([24, 53, 23, 25, 32, 52, 22, 43, 52, 48]), 'salary': pd.Series(
    [40000, 52000, 25000, 77000, 48000, 110000, 38000, 44000, 27000, 65000]),
                                 'college': pd.Series(['Y', 'N', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'Y'])})
print(dataq2)

Outcome = [1, 0, 0, 1, 1, 1, 1, 0, 0, 1]
dataq2['Outcome'] = Outcome

import math
import pandas as pd


class TreeNode(object):
    def __init__(self, attribute, threshold):
        self.attr = attribute
        self.thres = threshold
        self.left = None
        self.right = None
        self.leaf = False
        self.predict = None

# First select the threshold of the attribute
def select_threshold(df, attribute, outcome):
    # Convert dataframe column to a list and round each value
    values = df[attribute].tolist() 
    # Remove duplicate values, then sort the set
    values = list(set(values))
    values.sort()
    max_ig = float("-inf")
    thres_val = 0
    # try all threshold values that are half-way between successive values in this sorted list
    for i in range(0, len(values) - 1):
        thres = (values[i] + values[i + 1]) / 2
        ig = info_gain(df, attribute, outcome, thres)
        if ig > max_ig:
            max_ig = ig
            thres_val = thres
    # Return the threshold value that maximizes information gained
    return thres_val

#calculates the entropy of the data set provided
def info_entropy(df, outcome):
    # Dataframe and number of positive/negatives examples in the data
    p_df = df[df[outcome] == 1]
    n_df = df[df[outcome] == 0]
    p = float(p_df.shape[0])
    n = float(n_df.shape[0])
    # Calculate entropy
    if p == 0 or n == 0:
        I = 0
    else:
        I = ((-1 * p) / (p + n)) * math.log(p / (p + n), 2) + ((-1 * n) / (p + n)) * math.log(n / (p + n), 2)
    return I


# Calculates the weighted average of the entropy after an attribute test
def remainder(df, df_subsets, outcome):
    # number of test data
    num_data = df.shape[0]
    remainder = float(0)
    for df_sub in df_subsets:
        if df_sub.shape[0] > 1:
            remainder += float(df_sub.shape[0] / num_data) * info_entropy(df_sub, outcome)
    return remainder


# Calculates the information gain from the attribute test based on a given threshold
# Note: thresholds can change for the same attribute over time
def info_gain(df, attribute, outcome, threshold):
    sub_1 = df[df[attribute] < threshold]
    sub_2 = df[df[attribute] > threshold]
    # Determine information content, and subract remainder of attributes from it
    ig = info_entropy(df, outcome) - remainder(df, [sub_1, sub_2], outcome)
    return ig


# Returns the number of positive and negative data
def num_class(df, outcome):
    p_df = df[df[outcome] == 1]
    n_df = df[df[outcome] == 0]
    return p_df.shape[0], n_df.shape[0]

# Chooses the attribute and its threshold with the highest info gain
# from the set of attributes
def choose_attr(df, attributes, outcome):
    max_info_gain = float("-inf")
    best_attr = None
    threshold = 0
    # Test each attribute (note attributes maybe be chosen more than once)
    for attr in attributes:
        thres = select_threshold(df, attr, outcome)
        ig = info_gain(df, attr, outcome, thres)
        if ig > max_info_gain:
            max_info_gain = ig
            best_attr = attr
            threshold = thres
    print(best_attr,threshold,max_info_gain)
    return best_attr, threshold


# Builds the Decision Tree based on training data, attributes to train on,
# and a prediction attribute
def build_tree(df, cols, outcome):
    # Get the number of positive and negative examples in the training data
    p, n = num_class(df, outcome)
    # If train data has all positive or all negative values
    # then we have reached the end of our tree
    if p == 0 or n == 0:
        # Create a leaf node indicating it's prediction
        leaf = TreeNode(None, None)
        leaf.leaf = True
        if p > n:
            leaf.predict = 1
        else:
            leaf.predict = 0
        return leaf
    else:
        # Determine attribute and its threshold value with the highest
        # information gain
        best_attr, threshold = choose_attr(df, cols, outcome)
        # Create internal tree node based on attribute and it's threshold
        tree = TreeNode(best_attr, threshold)
        sub_1 = df[df[best_attr] < threshold]
        sub_2 = df[df[best_attr] > threshold]
        # Recursively build left and right subtree
        tree.left = build_tree(sub_1, cols, outcome)
        tree.right = build_tree(sub_2, cols, outcome)
        return tree


# Given a instance of a training data, make a prediction of healthy or colic
# based on the Decision Tree
# Assumes all data has been cleaned (i.e. no NULL data)
def predict(node, row_df):
    # If we are at a leaf node, return the prediction of the leaf node
    if node.leaf:
        return node.predict
    # Traverse left or right subtree based on instance's data
    if row_df[node.attr] <= node.thres:
        return predict(node.left, row_df)
    elif row_df[node.attr] > node.thres:
        return predict(node.right, row_df)


# Given a set of data, make a prediction for each instance using the Decision Tree
def test_predictions(root, df):
    num_data = df.shape[0]
    num_correct = 0
    for index, row in df.iterrows():
        prediction = predict(root, row)
        if prediction == row['Outcome']:
            num_correct += 1
    return round(num_correct / num_data, 2)

def main():
    # An example use of 'build_tree' and 'predict'
    attributes = ['age', 'salary']
    root = build_tree(dataq2, attributes, 'Outcome')
    print("Accuracy of test data")
    print(str(test_predictions(root, dataq2) * 100.0) + '%')

if __name__ == '__main__':
    main()



#number 3 part 1

#122 total
#5 * (4*4) * (3*3*3) = 6,480

#Part b
#product from 

#part c
#deeper trees have higher variance.

#part e