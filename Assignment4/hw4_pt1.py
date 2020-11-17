​
import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
​
​
​
def RBFKernel(X1,X2,sigma,h):
    #bonus if implemented without for loops. no thanks im good.
    K = np.zeros((X1.shape[0],X2.shape[0]))
    for i in range(X1.shape[0]): 
        for j in range(X2.shape[0]):
            K[i,j] = sigma*np.exp(((-np.linalg.norm(X1[i]-X2[j])**2)/2*(h**2)))
    return K
​
​
def GPRegression(XTrain, yTrain, XTest, gamma, sigma, h):
    n = np.zeros(XTrain.shape[0])
    ym = np.mean(yTrain) 
    y = yTrain - ym
    K = RBFKernel(XTrain, XTrain, sigma,h)
    L_ = gamma*np.eye(len(n))
    L_n = K + (np.tril(L_,k=0))
    L = np.linalg.cholesky(L_n)
    #mean
    #alpha = np.linalg.inv(L.T)*(np.linalg.inv(L)*y)
    alpha = np.linalg.solve(L.T,np.linalg.solve(L,y))
    GPMean = ym + ((K.T)*alpha)
    #variance
​
    v = np.linalg.solve(L,K)
    GPVariance = K - ((v.T)*v)
​
    return GPMean, GPVariance
​
def LogMarginalLikelihood(XTrain,yTrain,gamma,sigma,h):
    n = np.zeros(XTrain.shape[0],dtype=int)
    
    ym = np.mean(yTrain)
    #print(ym)
    y = yTrain - ym
    K = RBFKernel(XTrain,XTrain,sigma,h)
​
    L_ = gamma*np.eye(len(n))
    L_n = K + (np.tril(L_,k=0))
    L = np.linalg.cholesky(L_n)
​
    alpha = np.linalg.solve(L.T,np.linalg.solve(L,y))
    logml = np.mean((-1/2*y.T*alpha)-(sum(np.log(np.diag(L))) - ((n/2)*np.log(2*np.pi))))
    return logml,sigma,h
​
​
def HyperParameters(XTrain,yTrain,hs,sigmas):
    #here you can just call the grid search function provided by Python
    gamma = 0.01*(np.std(yTrain))
    #this does a grid search across the parameters in hs, sigmas and returns the combination that minimizes the log marginal likelihood 
​
    margs_list = []
    for i in sigmas:
        for j in hs:
            lm,s,h = LogMarginalLikelihood(XTrain,yTrain,gamma,i,j)
            margs_list.append(lm)
            if np.min(margs_list) == lm:
                new_min = [lm,s,h]
            else:
                pass
    h = new_min[2]
    sigma = new_min[1]
    return gamma,h,sigma
​
            
​
data = pd.read_excel('/Users/lvbenson/Research_Projects/MachineLearning/Assignment4/Concrete_Data.xls')
X = data.iloc[:,:-1] # Features 
y = data.iloc[:,-1] # Target 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2) 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)
​
#logspace(-1,1,10)’?norm(std(XTrain)) 
hs_ = (np.logspace(-1,1,num=10).T)*np.linalg.norm(np.std(X_train))
​
sigmas_ = (np.logspace(-1,1,num=10).T)*(np.std(y_train))
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
​
gamma = 0.01*(np.std(y_train))
g,h,s = HyperParameters(X_train,y_train,hs_,sigmas_)
M,V = GPRegression(X_train, y_train, X_test, g, s, h)
print(M.shape,'mean')
print(V.shape,'variance')