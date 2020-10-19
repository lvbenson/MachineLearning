import os
import pandas as pd
import numpy as np 
import scipy.stats #for the pdf function 
from sklearn.linear_model import Ridge

offset = 2
#p_x_1 = scipy.stats.norm(loc=[0,0], scale=[[1,0],[0,1]])
#p_x_2 = scipy.stats.norm(loc[offset,offset], scale=[[1,0],[0,1]])


def toydata(n):
    class_1_size = n/2
    class_2_size = n - class_1_size
    classes = {}

    for example1,example2 in zip(range(int(class_1_size)),range(int(class_2_size))):
        point1 = np.random.normal(loc=[0,0], scale=[[1,0],[0,1]])
        x1 = point1[0,0]
        y1 = point1[1,1]
        classes[(x1,y1)] = 1
        point2 = np.random.normal(loc=[offset,offset], scale=[[1,0],[0,1]])
        x2 = point2[0,0]
        y2 = point2[1,1]
        classes[(x2,y2)] = -1

    #print(classes)
    return classes


"""
def computeybar(example):
    #p_x_y = p_x_1.pdf(example)*p_x_2.pdf(example) 
    pass
"""
#points = toydata(500)

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
computehbar(points)


def computevariance(hbar):

    #25 Hds, applied to all 25 vectors

    sub_var = []
    for m in classifications:
        sub = m - hbar
        sub_var.append(sub**2)

    variance = np.mean(sub_var)
    return variance







    
    
    



    
