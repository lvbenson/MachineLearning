
import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_excel('/Users/lvbenson/Research_Projects/MachineLearning/Assignment4/Concrete_Data.xls')
X = data.iloc[:,:-1] # Features 
y = data.iloc[:,-1] # Target 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2) 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)
#print(len(X_train)) #772
#print(len(X_test)) #258



def RBFKernel(X1,X2,sigma,h):
    #bonus if implemented without for loops. no thanks im good.
    K = np.zeros((X1.shape[0],X2.shape[0]))
    for i in range(X1.shape[0]): 
        for j in range(X2.shape[0]):
            K[i,j] = sigma*(np.exp((-np.linalg.norm(X1[i]-X2[j])**2)/2*h**2))
    return K


def GPRegression(XTrain, yTrain, XTest, gamma, sigma, h):
    n = np.zeros(XTrain.shape[0])
    ym = np.mean(yTrain) 
    y = yTrain - ym
    K = RBFKernel(XTrain, XTest, sigma,h)
    L = np.linalg.cholesky(K + (gamma)*np.identity(n))
    #mean
    alpha = np.linalg.inv(L.T)*(np.linalg.inv(L)*y)
    GPMean = ym + ((K.T)*alpha)
    #variance
    v = np.linalg.inv(L)*K
    GPVariance = K - ((v.T)*v)

    return GPMean, GPVariance

def LogMarginalLikelihood(XTrain,yTrain,gamma,sigma,h):
     #need for logml: alpha,L,ytrain,ym,K. basically everything as in GPRegression
    n = np.zeros(XTrain.shape[0])
    ym = np.mean(yTrain)
    y = yTrain - ym
    K = RBFKernel(XTrain,XTrain, sigma,h)
    L = np.linalg.cholesky(K + (gamma)*np.identity(n)) 
    alpha = np.linalg.inv(L.T)*(np.linalg.inv(L)*y)
    logml = -1/2*y.T*alpha-sum(np.log(np.diag(L))) - ((n/2)*np.log(2*np.pi))
    return logml

def HyperParameters(XTrain,yTrain,hs,sigmas):
    #here you can just call the grid search function provided by Python
    gamma = 0.01*(np.std(yTrain))
    #this does a grid search across the parameters in hs, sigmas and returns the combination that minimizes the log marginal likelihood 
    param_grid = dict(hs=hs,sigmas=sigmas)
    #i think i do a gridsearchCV over the LogMarginalLikelihood function
    log_ml = LogMarginalLikelihood(XTrain,yTrain,gamma,sigmas,hs) #????
    grid = GridSearchCV(estimator = log_ml, param_grid=param_grid, n_jobs=-1, cv=3) 
    
    #return gamma,h,sigma
    