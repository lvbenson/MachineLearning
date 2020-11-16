import os
import pandas as pd
import numpy as np 
import scipy
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt

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

    return classes


def computeybar(points): 
    prob_x_1 = scipy.stats.multivariate_normal(mean=[0,0], cov=np.identity(2)) #class1
    prob_x_2 = scipy.stats.multivariate_normal(mean=[offset,offset], cov=np.identity(2)) #class2
    ybar = []
    prob_y_1 = 0.5
    prob_y_2 = 0.5

    for x in points:
        p_d_f = (-prob_x_1.pdf(x)*prob_y_1 + prob_x_2.pdf(x)*prob_y_2)/(prob_x_1.pdf(x)*prob_y_1 + prob_x_2.pdf(x)*prob_y_2)
        ybar.append(p_d_f)
    
    #print(np.array(ybar))
    y_bar = np.array(ybar)
    #print(y_bar.shape)
    return y_bar

def computehbar(lambda_,points,num_models=25): #points as calculated from toydata
    #generate nmodel many models (Ridge regression)
    models = []
    for mod in range(num_models):
        ridge_mod = Ridge(alpha=10**lambda_)
        models.append(ridge_mod)

    #models = [Ridge(alpha=10**lambda_) for _ in range(num_models)]

    #generate n-many training sets for these n-many models
    datasets = []
    for mod in range(num_models):
        dat = toydata(500)
        datasets.append(dat)

    #datasets = [toydata(500) for _ in range(num_models)]

    #train the n-many models
    for i in range(num_models):
        X = np.array(list(datasets[i].keys()))
        y = np.array(list(datasets[i].values()))
        models[i].fit(X=X,y=y)

    classifications = []
    for mod in models:
        m_ = np.array(list(points.keys()))
        m_pred = mod.predict(m_)
        classifications.append(m_pred)


    #classifications = np.array([m.predict(np.array(list(points.keys()))) for m in models])
    #25 models trained over each of the 500 points 
    #across rows, different predictions for the same point

    #mean of the classifications, for each of the 500 points
    #for each coordinate, in 500x1, it is the 25 model predictions for that X (where the X is the point in question)
    hbar = np.mean(classifications, axis=0)
    return hbar,classifications


def computevariance(hbar,classifications):
    #25 Hds, applied to all 25 vectors

    sub_var = []
    for m in classifications:
        sub = m - hbar
        sub_var.append(sub**2)
    variance = np.mean(sub_var)
    return variance


def computeBias(hbar,ybar):
    bias = np.mean((hbar-ybar)**2)
    return np.mean(bias)


def computeNoise(ybar,Y):
    noise = np.mean((ybar-Y)**2)
    return noise
    
def biasvariancedemo():
    #plot: variance, bias, noise, test error, bias+variance+noise

    var_list = []
    bias_list = []
    noise_list = []
    error_list = []

    for lambda_ in np.arange(-10,10,0.1):
        points = toydata(500)
        X = np.array(list(points.keys()))
        Y = np.array(list(points.values()))
        
        #get ybar
        ybar_ = computeybar(points)
        #get hbar
        hbar_,classifications_ = computehbar(lambda_,points)
        #get variance
        var = computevariance(hbar_,classifications_)
        var_list.append(var)
        #get bias
        bias_ = computeBias(hbar_,ybar_)
        bias_list.append(bias_)
        #get noise
        noise_ = computeNoise(ybar_,Y)
        noise_list.append(noise_)
        #get error
        error = var + bias_ + noise_
        error_list.append(error)
    return var_list,bias_list,noise_list,error_list

v,b,n,e = biasvariancedemo()

b_v = np.add(b,v)  
b_v_n = np.add(b_v,n)
plt.plot(v,label='variance')
plt.plot(b,label='bias')
plt.plot(n,label='noise')
plt.plot(e,label='error')
plt.plot(b_v_n,label='bias+var+noise')
plt.legend()
plt.show()
    












    
