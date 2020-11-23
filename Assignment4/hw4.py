#!/usr/bin/env python
# coding: utf-8

# # q1

# In[ ]:

'''
## for1.1., I looked at the online resources here http://krasserm.github.io/2018/03/19/gaussian-processes/, and here https://towardsdatascience.com/understanding-gaussian-process-the-socratic-way-ba02369d804, and here https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution

## for 1.2., I referred to codes here: https://katbailey.github.io/post/gaussian-processes-for-dummies/ and http://krasserm.github.io/2018/03/19/gaussian-processes/


# 1.1  Lemma

# 
# note that all conditional distributions of a multivariate normal distribution are normal
# 
# Define z=y∗+Ay, whereA=−Σ12 Σ22 -1=−K(X∗,X)K(X,X)−1 
# 
# Now we can write cov(z,x2)=cov(x1,x2)+cov(Ax2,x2)=K(X∗,X)+Avar(x2)=K(X∗,X)−K(X∗,X)K(X,X)-1K(X,X)=0
# Thus z,y are independent 
# note : all the -1 is on the up right corner but I dont know how to write it in the correct format in notebook
# 

# 
# we can derive the mean from: 
# E(y*|y)=E(z−Ay|y)=E(z|y)−E(Ay|y)=E(z)−Ay=−Ay=−K(X∗,X)K(X,X)−1y
# 
# 
# The variance is:
# Var(y∗|y)
# =Var(z−Ay|y)
# =var(z|y)+var(Ay|y)−Acov(z,−y)−cov(z,−y)A′
# =var(z|y)
# =var(z)
# =Var(y∗+Ay)
# =Var(y∗) +Var(Ay)−Cov(y∗,Ay)−Cov(Ay,y∗)
# =K(X∗,X∗) +AK(X,X)AT−Cov(y∗,y)AT−ACov(y,y∗)
# =K(X∗,X∗) +AK(X,X)AT−K(X∗,X)AT−AK(X,X∗)
# =K(X∗,X∗)−K(X∗,X)K(X,X)−1K(X,X∗)
# 
# note : all the -1 and T is on the up right corner but I dont know how to write it in the correct format in notebook
# 

# 1.2 Gaussian Progress Regression

# In[36]:


import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel ('Concrete_Data.xls', sheet_name='Sheet1')
print (df.head())
print(df.columns)

# split data
y=df[['Concrete compressive strength(MPa, megapascals) ']]
X= df.drop(['Concrete compressive strength(MPa, megapascals) '], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
N=X_train.shape[0]
print(N)


# In[38]:


# Define the rbf kernel function
def kernel(x1,x2, sigma, h):
    sqdist = np.sum(x1**2,1).reshape(-1,1) + np.sum(x2**2,1) - 2*np.dot(x1, x2.T)
    expo=np.exp(-.5 * sqdist/h**2)
    return sigma * expo


# In[40]:


# compute the mean and variances at our test points
def GPRegression(XTrain, yTrain, XTest, sigma,h): 

    n1 = 100
    n2 = 80
    X1=X_train.sample(frac=n1/N, random_state=1)
    X2=X_train.sample(frac=n2/N, random_state=2)
    print (X1.shape)
    print (X2.shape)

    K_rbf = kernel(X1.values.reshape(-1,1), X2.values.reshape(-1,1),1,1)
    print (K_rbf)
    print (K_rbf.shape)

    # compute the mean at our test points.
    L = np.linalg.cholesky(K_rbf + np.eye(N))
    Lk = np.linalg.solve(L, kernel(XTrain, XTest))
    mu = np.dot(Lk.T, np.linalg.solve(L, yTrain))

    # compute the (co)variance at our test points.
    K_ = kernel(Xtest, Xtest)
    s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
    s = np.sqrt(s2)
    return mu,s


# In[ ]:


# parameters optimization
import scipy
from scipy.optimize import minimize

def LogMarginalLikelihood(XTrain, yTrain, gamma, sigma, h,naive=False): 
        """
    Returns a function that computes the negative log marginal
    likelihood for training data X_train and Y_train and given
    noise level.

    Args:
        X_train: training locations (m x d).
        Y_train: training targets (m x 1).
        noise: known noise level of Y_train.
        naive: if True use a naive implementation of Eq. (11), if
               False use a numerically more stable implementation.

    Returns:
        Minimization objective.
    """
    
    Y_train = yTrain.ravel()
        
    def llnormal(theta):
        # more stable implementation of Eq. (11) as described in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section 2.2, Algorithm 2.1.
        
        def ls(a, b):
            return numpy.linalg.lstsq(a, b, rcond=-1)[0]
        
        K = kernel(XTrain, XTrain, h, sigma) + gamma**2 * np.eye(len(X_train))
        L = numpy.linalg.cholesky(K)
        return np.sum(np.log(np.diagonal(L))) + 0.5 * Y_train.dot(ls(L.T, ls(L, Y_train))) + 0.5 * len(XTrain) * np.log(2*np.pi)
    
    def llnaive(theta):
        # Naive implementation to answer the last subquestion
        K = kernel(XTrain, XTrain, h, sigma) + gamma**2 * np.eye(len(XTrain))
        return 0.5 * np.log(numpy.linalg.det(K)) + 0.5 * Y_train.dot(numpy.linalg.inv(K).dot(Y_train)) + 0.5 * len(XTrain) * np.log(2*np.pi)
    if naive:
        return llnaive
    else:
        return llnormal

    # Minimize the negative log-likelihood
    logml = minimize(LogMarginalLikelihood(XTrain, yTrain, gamma), [1, 1], 
               bounds=((1e-5, None), (1e-5, None)),
               method='L-BFGS-B')
    return logmal


# #gridsearch
# from sklearn.model_selection import ParameterGrid
# logml=0
# def HyperParameters(XTrain, yTrain, hs, sigmas):
#     param_grid = {'sigma':sigmas,'h': hs}
#     print(list(ParameterGrid(param_grid)))
#     for i in list(ParameterGrid(param_grid)):
#         print(i)
#         if LogMarginalLikelihood(XTrain, yTrain, gamma, i.get('sigma'), i.get('h'),naive=False)>logml:
#             logml=LogMarginalLikelihood(XTrain, yTrain, gamma, i.get('sigma'), i.get('h'),naive=False)
#             print(sigma,h,logml)
#     return h,sigma

# In[ ]:


hs = logspace(-1,1,10).T*norm(std(XTrain))
sigmas = logspace(-1,1,10).T* std(yTrain)
h, sigma =HyperParameters(XTrain, yTrain, hs, sigmas)  
gamma = 0.01 * GPRegression(XTrain, yTrain, XTest, sigma,h)[1]

l_opt, sigma_f_opt = res.x

# Compute posterior mean and covariance with optimized kernel parameters and plot the results
mu_s, cov_s = posterior(X, X_train, Y_train, h, sigma, sigma_y=noise)
plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
'''

# # q2

# In[ ]:

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


    print('yo whaddup my dudes! its ya boi, cowboy dan dan noodle, back at it again with proofs and depression')