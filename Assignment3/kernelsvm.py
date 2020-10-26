import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import math

#create dataset of points
#f=open('/Users/benso/Desktop/Projects/MachineLearning/Assignment3/hw3_data2.txt',"r")
f=open('/Users/lvbenson/Research_Projects/MachineLearning/Assignment3/hw3_data2.txt',"r")

lines = f.readlines()
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

#f=open('/Users/benso/Desktop/Projects/MachineLearning/Assignment3/hw3_data2.txt',"r")
f=open('/Users/lvbenson/Research_Projects/MachineLearning/Assignment3/hw3_data2.txt',"r")
lines1=f.readlines()
result2=[]
for x in lines1:
    result2.append(x.split(',')[1])
f.close()

x_coords = []
for coord in result2:
    x_coords.append(coord)

#f=open('/Users/benso/Desktop/Projects/MachineLearning/Assignment3/hw3_data2.txt',"r")
f=open('/Users/lvbenson/Research_Projects/MachineLearning/Assignment3/hw3_data2.txt',"r")
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
f.close()

X = np.array(data,dtype=float)

#create test, train vectors
x_train,y_train,x_test,y_test = train_test_split(X,target)

#3600 data points
#we have coordinates for our X (training) data, and a target vector

def rbf_kernel(X,Y,gamma):
    K = np.zeros((X.shape[0],X.shape[0]))
    #print(Y.shape[0])
    for i in range(X.shape[0]): 
        for j in range(X.shape[0]):
            K[i,j] = np.exp(-gamma*np.linalg.norm(X[i]-X[j])**2) # rbf alg, kinda like euclid distance
    return K


def CalcCost(w,x,kernel,classification=1):
    #loss func, non-linear svm
    #cost1: 
    if classification == 1:
        return max(0, 1 - w.T.dot(x))
    elif classification == -1:
        return max(0, 1 + w.T.dot(x))
    
#c is some parameter in the loss function
def CalcLoss(w,K,X,Y,c=1,reg=1):
    loss = 0
    for i in range(len(y)):
        f_i = np.array([K[i,k] for k in range(K.shape[0])])
        c_1 = CalcCost(f_i,w,1)
        c_0 = CalcCost(f_i,w,-1)

        loss += y[i]*c_1 + (1-y[i])*c_0
    
    loss = c*loss
    loss = reg*w.T.dot(w)

    if -y[i]*w.T.dot(x[i]) < 1:
        L = [i for i in range(len(y))]
    
    dw = np.array([
        -y[i] * sum([y[j]*K[j,i] for j in L]) for i in range(len(y))
    ])


    return loss, dw

def SVM(X,Y,K,epochs=1000,rate=1,reg=1,C=1,gamma=1):
    #initialize random weights
    w = np.zeros_like(X[0])
    for epoch in epochs:
        list_of_loss = []
        
        loss,dw = CalcLoss(w,K,X,Y,reg,C)
        list_of_loss.append(loss)

        w = np.subtract(w,dw)
    
    return w












"""

def predict_sgd(Xi,X,kernelFunction,w,b,sigma):
    K=kernelFunction(Xi, X, sigma)
    for i in range(len(L)):
        if np.dot(w, K[i])+b >= 1:
            return 1
        else:
            return 0


def classify(X,w,b,Y,kernel):

    f_x = 0
    for i in range(len(Y)):
        f_x += w[i]*Y[i]*kernel[i_x,i]+b
    
    return f_x


def LossFunc(X,Y,w,kernel):

    w_dot_x = X.dot(w.T)
    s_yi = w_dot_x[np.arange(x.shape[0]),y]

    return s_yi

"""











"""
    #return K #this gives the covariance kinda
    epochs = 5
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
            acc += a*y*np.exp(-gamma*np.linalg.norm(x-a)**2) #check validity of alpha
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

"""