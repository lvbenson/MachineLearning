
import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import GridSearchCV

def RBFKernel(X1,X2,sigma,h):
    K = np.zeros((X1.shape[0],X2.shape[0]))
    for i in range(X1.shape[0]): 
        for j in range(X2.shape[0]):
            K[i,j] = sigma*(np.exp((-gamma*np.linalg.norm(X1[i]-X2[j])**2)/2*h**2))
    return K


def GPRegression(XTrain, yTrain, XTest, gamma, sigma, h):
    n = np.zeros(Xtrain.shape[0])
    ym = np.mean(yTrain)
    y = yTrain - ym
    K11 = RBFKernel_T(XTrain,Xtrain,sigma,h)
    K12 = RBFKernel_T(XTrain, XTest, sigma,h)
    K22 = RBFKernel_T(XTest, XTest, sigma,h)
    L = np.linalg(K11 + (gamma)*np.identity(n))
    #mean
    alpha = (L.T)/(L/y)
    GPMean = ym + ((K12.T)*alpha)
    #variance
    v = L/K12
    GPVariance = K22 - ((v.T)*v)

    return GPMean, GPVariance

def LogMarginalLikelihood(XTrain,yTrain,gamma,sigma,h):
    n = np.zeros(XTrain.shape[0])
    ym = np.mean(yTrain)
    y = yTrain - ym
    K11 = RBFKernel_T(XTrain,XTrain, sigma,h)
    L = np.linalg(K11 + (gamma)*np.identity(n))
    alpha = L.T/(L/y)
    logml = -1/2*y.T*alpha-sum(np.log(np.diag(L))) - ((n/2)*np.log(2*np.pi))
    return logml

def HyperParameters(XTrain,yTrain,hs,sigmas):
    #here you can just call the grid search function provided by Python
    gamma = 0.01*(np.std(yTrain))
    #this does a grid search across the parameters in hs, sigmas and returns the combination that minimizes the log marginal likelihood 
    params = 
    




function [gamma, h,sigma] = HyperParameters(XTrain, yTrain,hs,sigmas)
  nsearch = numel(hs);
  params = [repmat(hs,nsearch,1) repmat(sigmas,1,nsearch)'(:)];
    
  gamma = 0.01*std(yTrain);
  best = -inf;
  for i=1:size(params,1)
    cur_h = params(i,1);
    cur_sigma = params(i,2);
    [logml] = LogMarginalLikelihood_T (XTrain, yTrain, gamma, cur_sigma,cur_h);
    if logml > best
      best = logml;
      h = cur_h;
      sigma = cur_sigma;
    end
  end
endfunction