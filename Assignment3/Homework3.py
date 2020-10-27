import os
import pandas as pd
import numpy as np 
import scipy
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import Ridge

offset = 2

def toydata(n):
    class_1_size = n/2
    class_2_size = n - class_1_size
    classes = {}

    for example1,example2 in zip(range(int(class_1_size)),range(int(class_2_size))):
        #draw random points from normal distribution
        point1 = np.random.normal(loc=[0,0], scale=[1,1])
        (x1,y1) = point1[0],point1[1]
        classes[(x1,y1)] = -1 #put into class1
        point2 = np.random.normal(loc=[offset,offset], scale=[1,1])
        (x2,y2) = point2[0],point2[1]
        classes[(x2,y2)] = 1 #put into class2

    #print(classes)
    return classes


def computeybar(): 
    prob_x_1 = scipy.stats.norm(loc=[0,0], scale=[1,1]) #class1
    prob_x_2 = scipy.stats.norm(loc=[offset,offset], scale=[1,1]) #class2
    ybar = []
    prob_y_1 = 0.5
    prob_y_2 = 0.5

    for x in toydata(500):
        #print(x)
        denom = prob_x_1.pdf(x)[0]*prob_y_1 + prob_x_2.pdf(x)[1]*prob_y_2
        num = -prob_x_1.pdf(x)[0]*prob_y_1 + prob_x_2.pdf(x)[1]*prob_y_2
        div = num/denom
        ybar.append(div)
    
    #print(np.array(ybar))
    return np.array(ybar)

    #pdf(0,1) * (0.5) / 
    #ybar(x) = -P(x|y=1)P(y=1)/P(x|y=1)P(y=1) + P(x|y=2)P(y=2) +
    #P(x|y=2)P(y=2)/P(x|y=1)P(y=1) + P(x|y=2)P(y=2)

    
#points = toydata(500)
y_bar = computeybar()

def computehbar(points, num_models=25): #points as calculated from toydata
    #generate nmodel many models (Ridge regression)
    models = [Ridge() for _ in range(num_models)]

    #generate n-many training sets for these n-many models
    datasets = [toydata(500) for _ in range(num_models)]
    #print(datasets)

    #train the n-many models
    for i in range(num_models):
        X = np.array(list(datasets[i].keys()))
        y = np.array(list(datasets[i].values()))
        models[i].fit(X=X,y=y)
        #models[i].fit(X=np.array(datasets[i].keys()), y=datasets[i].values())

    #get the predictions from training corpus
    #takes point value, returns which class it is in 

    #
    classifications = np.array([m.predict(np.array(list(points.keys()))) for m in models])
    #25 models trained over each of the 500 points 
    #across rows, different predictions for the same point

    #mean of the classifications, for each of the 500 points
    #for each coordinate, in 500x1, it is the 25 model predictions for that X (where the X is the point in question)
    hbar = np.mean(classifications, axis=0)
    #print(hbar)
    return hbar, models, classifications

points = toydata(500)
h,m,c = computehbar(points)
#print(h)


def computevariance(hbar,classifications):

    #25 Hds, applied to all 25 vectors

    sub_var = []
    for m in classifications:
        sub = m - hbar
        sub_var.append(sub**2)

    variance = np.mean(sub_var)
    #print(variance,'variance')
    return variance

var = computevariance(h,c)


def computeBias(hbar,ybar):
    bias = np.mean((hbar-ybar)**2)
    #print(bias,'bias')
    return bias
bias_ = computeBias(h,y_bar)


def computeNoise(ybar,Y):
    noise = np.mean((ybar-Y)**2)
    #print(noise,'noise')
    return noise

noise_ = computeNoise(y_bar,c)

Error = var + bias_ + noise_
#print(Error)

print('variance:',var,'bias:',bias_,'noise:',noise_,'Error:',Error)
    
    
    



    
