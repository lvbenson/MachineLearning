import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import math

#create dataset of points
f=open('/Users/benso/Desktop/Projects/MachineLearning/Assignment3/hw3_data2.txt',"r")

lines=f.readlines()
result=[]
for x in lines:
    result.append(x.split(',')[0])
f.close()

target = []
for sign in result:
    if sign == '+':
        target.append(1)
    else:
        target.append(-1)

f=open('/Users/benso/Desktop/Projects/MachineLearning/Assignment3/hw3_data2.txt',"r")
lines1=f.readlines()
result2=[]
for x in lines1:
    result2.append(x.split(',')[1])
f.close()

x_coords = []
for coord in result2:
    x_coords.append(coord)

f=open('/Users/benso/Desktop/Projects/MachineLearning/Assignment3/hw3_data2.txt',"r")
lines2=f.readlines()
result3=[]
for x in lines2:
    x_ = x.split(',')[2].rstrip('\n')
    result3.append(x_)
y_coords = []
for coord in result3:
    y_coords.append(coord)
data = []
for x,y in zip(x_coords,y_coords):
    data.append([x,y])

X = np.array(data,dtype=float)
dataset = np.insert(X,2,-1,axis=1)
#print(dataset)

#3600 data points
#we have coordinates for our X (training) data, and a target vector

def rbf_kernel(X,Y,gamma):
    alpha = np.zeros(X.shape[0])
    K = np.zeros((X.shape[0],Y.shape[0]))
    for i,x in enumerate(X): 
        for j,y in enumerate(Y):
            K[i,j] = np.exp(-gamma*np.linalg.norm(x-y)**2) #actual rbf alg
    #print(K)
    #return K #this gives the actual covariance
    epochs = 100
    for epoch in range(epochs):
        size = X.shape[0]
        for i in range(size):
            sum = 0
            val = 0
            for j in range(size):
                val = alpha[j] * Y[j] * K[i,j]
                sum = sum + val
            if sum <= 0:
                val = -1
            elif sum > 0:
                val = 1
            if val != Y[i]:
                alpha[i] = alpha[i] + 1
    #check accuracy
    print(alpha)
    return alpha

dataset = np.array(dataset)
target = np.array(target)
matrix = rbf_kernel(dataset,target,1.0)

"""


def compute_efficiency(train_X,train_Y,test_X,test_Y,alpha):
    m = test_Y.size
    right = 0
    for i in range(m):
        s = 0
        for a, x_train,y_train  in zip(alpha, train_X,train_Y):
            s += a * y_train * rbf(test_X[i],x_train)
        if s >0:
            s = 1
        elif s <=0:
            s = -1
        if test_Y[i] == s:
            right +=1

    print " Correct : ",right," Accuracy : ",right*100/test_X.shape[0]
        #return y_predict

    """