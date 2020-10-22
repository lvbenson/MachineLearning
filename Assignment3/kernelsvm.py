import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sn

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

    K = np.zeros((X.shape[0],Y.shape[0]))
    for i,x in enumerate(X): 
        for j,y in enumerate(Y):
            K[i,j] = np.exp(-gamma*np.linalg.norm(x-y)**2) #actual rbf alg
    #print(K)
    return K #this gives the actual covariance

dataset = np.array(dataset)
target = np.array(target)
matrix = rbf_kernel(dataset,target,1.0)

#how do I test the accuracy?

#sn.heatmap(matrix, annot=True, fmt='g')
#plt.show()

"""
for d, sample in enumerate(dataset):
    if target[d] == -1:
        plt.scatter(sample[0], sample[1], s=100, marker='o', c='blue',linewidths=2)
    else:
        plt.scatter(sample[0], sample[1], s=100, marker='*', c='red',linewidths=2)

"""