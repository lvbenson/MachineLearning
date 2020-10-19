# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:16:33 2020

@author: Lauren Benson
"""
import random
import matplotlib.pyplot as plt
import numpy as np


#part b

#Perceptron: Given a training dataset,it automatically learns the optimal weight coefficients that are 
#then multiplied with the input features in order to make the decision of whether a neuron fires or not.

#binary classifier: the target is either 1 or -1. If result is > 0, classified as 1. < 0, classified as -1.
#at each iteration on a data point, if the classification is wrong, algorithm adjusts the weights. 

#y = class label: either -1 or 1
#x = data point under consideration. Taken from a randomly generated dataset of coordinates. 

  
    
def Perceptron(dataset,target):
    incorrect_classify = dataset
    correct_classify = []
    w = np.ones(dataset.shape[1])
    while list(incorrect_classify):
        index = np.random.choice(incorrect_classify.shape[0])  
        example = dataset[index]
        if np.dot(example, w) < 0 and target[index] == -1:
            correct_classify.append(example) #add to correct classify
            incorrect_classify = np.delete(incorrect_classify,index,0) #delete from incorrect classify
            
            
        elif np.dot(example, w) > 0 and target[index] == 1:
            correct_classify.append(example) #add to correct classify
            incorrect_classify = np.delete(incorrect_classify,index,0) #delete from incorrect classify
        else:
            w = w + example*target[index] #update weight
        #print(len(incorrect_classify))
    
    print(w)
    return(w)
        
        
        
X = np.array([
    [-2,4,1],
    [4,1,1],
    [1, 6, 1],
    [2, 4, 1],
    [6, 2, 1]])

Y = [1,-1,1,-1,1]

Perceptron(X,Y)
      



