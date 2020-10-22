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
    alpha = np.zeros(X.shape[0]) #actual vector to update
    K = np.zeros((X.shape[0],X.shape[0]))
    #print(Y.shape[0])
    for i in range(X.shape[0]): 
        for j in range(X.shape[0]):
            K[i,j] = np.exp(-gamma*np.linalg.norm(X[i]-X[j])**2) # rbf alg, kinda like euclid distance
    #print(K)
    #return K #this gives the covariance kinda
    epochs = 1
    for epoch in range(epochs):
        size = X.shape[0]
        for i in range(size):
            update = 0
            check = 0
            for j in range(size):
                check = alpha[j] * Y[j] * K[i,j] #kind of like the dot product thing before
                update = update + check
            if update <= 0:
                check = -1 #correctly classified
            elif update > 0: 
                check = 1 #correctly classified
            if check != Y[i]: #if incorrectly classified
                alpha[i] = alpha[i] + 1 #update alpha
    #check accuracy
    #target = np.array(target)

    correct = 0
    for i in range(Y.shape[0]):
        acc = 0
        for a,x,y in zip(alpha,X,Y):
            acc += a*y*np.exp(-gamma*np.linalg.norm(x-y)**2) #check validity of alpha
        if acc > 0:
            acc = 1
        elif acc <= 0:
            acc = -1
        if Y[i] == acc:
            correct += 1

    print("num correct: ",correct," percent accurate : ",correct*100/X.shape[0])


    return alpha

dataset = np.array(dataset)
target = np.array(target)
vector = rbf_kernel(dataset,target,0.5)

