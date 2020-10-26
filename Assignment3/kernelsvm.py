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
Y = np.array(target,dtype=float)
#create test, train vectors
x_train,y_train,x_test,y_test = train_test_split(X,Y)

#3600 data points
#we have coordinates for our X (training) data, and a target vector
#with a kernel, the feature space isnt necessary to estimate K
def rbf_kernel(X,gamma):
    K = np.zeros((X.shape[0],X.shape[0]))
    #print(Y.shape[0])
    for i in range(X.shape[0]): 
        for j in range(X.shape[0]):
            K[i,j] = np.exp(-gamma*np.linalg.norm(X[i]-X[j])**2) # rbf alg, kinda like euclid distance
    return K


def CostFunc(x,w,classification):
    #loss func, non-linear svm
    #cost 0:
    class_0 = max(0, 1 - w.T.dot(x))
    print(1 - w.T.dot(x))
    print(class_0)
    class_1 = max(0, 1 + w.T.dot(x))
    if classification == 1:
        return class_0
    elif classification == -1:
        return class_1    
    
#c is some parameter in the loss function
#need w, x, y, kernel calculated, C parameter, regularization term
def Loss_Calculation(w,K,X,Y,reg_term=1,C=1): #K is 2700 x 2700 matrix (for training data), x is 2700x2, w is 2700x1

    loss = 0
    for i in range(len(Y)):

        f_i = np.array([K[i,k] for k in range(K.shape[0])]) #size=2700x1,floats
        c_1 = CostFunc(f_i,w,1) #returns either 1 or -1
        c_0 = CostFunc(f_i,w,-1) #returns either 1 or -1

        loss += y[i]*c_1 + (1-y[i])*c_0 #y is float,
    
    loss = C*loss
    loss = reg_term*w.T.dot(w)

    #if -y[i]*w.T.dot(x[i]) < 1:
     #   L = [i for i in range(len(y))]
    
    L = [i for i in range(len(y)) if -y[i]*classification(i,w,y,K)<1]
    
    #we need to calculate the change in w, or how to wiggle our w vector
    #according to each classification
    dw = np.array([
        -y[i] * sum([y[j]*K[j,i] for j in L]) for i in range(len(y))
    ])


    return loss, dw


def classification(index_x,w,Y,K):
    #w is len(X)

    fx = 0
    for i in range(len(Y)):
        fx += Y[i]*K[index_x][i]*w[i]
    return fx

def SVM(X,Y,K,epochs=1,learn_rate=1,reg_term=1,C=1,Gamma=1):
    #initialize random weights
    #w = np.zeros_like(X[0])
    w = np.zeros(X.shape[0])
    for epoch in range(epochs):
        loss_list = []
        
        loss,dw = Loss_Calculation(w,K,X,Y,reg_term,C)
        loss_list.append(loss)

        w = np.subtract(w,dw)
    
    return w

    
def pipeline(X,Y,epochs=1,learn_rate=1,reg_term=1,C=1,Gamma=1):
    K = rbf_kernel(X,Gamma)

    w_vector = SVM(X,Y,K,epochs,learn_rate)

    #need to classify everything in X 
    #things to classify:
    classification = [classification(i,w_vector,Y,K) for i in range(len(x))]
    """
    size = range(len(X))

    classify_list = []
    for index_x in size:
        classify_list.append(classification(index_x,w_vector,Y,K,))
    """
    return classification,w_vector


overall_shits = pipeline(x_train,y_train)
print(np.sign(overall_shits))


