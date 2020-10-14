#perceptron didn't work. Lets try an SVM.


import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot as plt
import Perceptron_hw2 as pc



f=open('/Users/lvbenson/MachineLearning_Fall2020/Assignment2/hw2_data1.txt',"r")
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

f=open('/Users/lvbenson/MachineLearning_Fall2020/Assignment2/hw2_data1.txt',"r")
lines1=f.readlines()
result2=[]
for x in lines1:
    result2.append(x.split(',')[1])
f.close()

x_coords = []
for coord in result2:
    x_coords.append(coord)

f=open('/Users/lvbenson/MachineLearning_Fall2020/Assignment2/hw2_data1.txt',"r")
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

def svm_sgd(X, Y):

    w = np.ones(len(X[0]))
    learning = 1.0
    epochs = 1100


    for epoch in range(1,epochs):
        for i, x in enumerate(X):
            if (Y[i]*np.dot(x, w)) < 1:
                w = w + learning * ( (x * Y[i]) + (-2*(1/epoch)* w) )
            else:
                w = w + learning * (-2*(1/epoch)* w)

    return w

w = svm_sgd(dataset,target)
print(w)

for d, sample in enumerate(dataset):
    if target[d] == -1:
        plt.scatter(sample[0], sample[1], s=100, marker='o', c='blue',linewidths=2)
    else:
        plt.scatter(sample[0], sample[1], s=100, marker='*', c='green',linewidths=2)

w2 = pc.Perceptron(dataset,target)
NewX = np.arange(0,100)
plt.plot(NewX, (-w[0]/w[1])*NewX - (w[2]/w[1]), c='red',label='hyperplane:SVM') #plot boundary line
plt.plot(NewX, (-w2[0]/w2[1])*NewX - (w2[2]/w2[1]), c='orange',label='hyperplane:Perceptron') #plot boundary line
plt.title('SVM vs Perceptron')
plt.legend()
plt.show()