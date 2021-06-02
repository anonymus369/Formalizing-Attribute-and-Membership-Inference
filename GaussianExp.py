#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import math
from numpy import linalg as la
from numpy import random as rd
from scipy.stats import multivariate_normal as MG
from scipy.stats import norm

# Draw X

def DrawX(d,n,sigmaX):
    """ Draw n feature vectors of leangth d.
    All feature vectors and componentes are drawn i.i.d. from a zero-centered normal distribution with std sigmaX.
    """
    return rd.normal(0,sigmaX,(n,d))

# Draw Y given X

def DrawY(X,beta,sigmaY):
    """ Draw scalar response Y according to X. 
    The dimensions of X must be [n,d], while beta must be a vector of length d.
    The added noise is generated from a zero-centered normal distribution with std sigmaY.
    """
    n = X.shape[0]
    Y = X@beta + rd.normal(0,sigmaY,n)
    return Y

# Mean Squared Error

def MSE(A,B):
    """ Compute the mean squared error between vectors A and B.
    A and B must be vectors of the same length.
    """
    return np.mean((A-B)**2)

# Train linear model

def TrainLinear(X,Y,Qinv):
    """Compute the optimum parameters to fit training set (X,Y).
    Qinv must be the inverse of np.transpose(X)@X"""
    return Y@X@Qinv

class SamplerTheta():
    """P.d.f. of the model parameters given hypothesis T and test sample S.
    X must be the matrix of feature vectors.
    Qinv = la.inv(np.transpose(X)@X) is pre-computed for efficiency.
    beta and sigmaY are the parameters of the p.d.f. of Y.
    """
    def __init__(self,T,beta,Qinv,sigmaY,X,S=None,i=None):
        if T==1 and ((i is None) or (S is None)):
            raise ValueError("If T is 1, an index and a scalar response must be provided")
        if T:
            x1 = X[i,:]
            y1 = S
            self.mean = beta + Qinv@x1*y1 - Qinv@x1*(x1@beta)
            self.cov = (sigmaY**2)*(Qinv-
                                 Qinv@x1[:,np.newaxis]@x1[np.newaxis,:]@Qinv)
        else:
            self.mean = beta
            self.cov = (sigmaY**2)*Qinv
        self.PDF = MG(mean=self.mean,cov=self.cov)
    
    def pdf(self,samples):
        """ Evaluate the density function at a given point.
        """
        return self.PDF.pdf(samples)
        
    def draw(self,n):
        """ Generate new samples.
        """
        return rd.multivariate_normal(self.mean,self.cov,n)
    
class SamplerS():
    """P.d.f. of scalar responses given feature vectors.
    beta and sigmaY are parameters of the p.d.f. of Y.
    """
    def __init__(self,beta,sigmaY,X):
        self.mean = X@beta
        self.cov = sigmaY
    
    def pdf(self,samples,i):
        """ Evaluate the density function at a given point.
        """
        PDF = norm(loc=self.mean[i],scale=self.cov)
        return PDF.pdf(samples)

def PSucLB(sigma,n,d,param):
    """ Compute the lower bound given by Theorem 2 in the paper.
    param corrersponds to Rmax in the paper.
    """
    sigma = sigma**2
    return .5 + (d*sigma)/(2*n*param) - math.exp(-param/(2*sigma))*(1+(2*sigma)/param)

# Golden Ratio

gr = (math.sqrt(5) + 1) / 2

def gss(f, a, b, tol=1e-5):
    """Golden-section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]

    Example:
    >>> f = lambda x: (x-2)**2
    >>> x = gss(f, 1, 5)
    >>> print("%.15f" % x)
    2.000009644875678

    """
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if f(c) > f(d):
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (a,b)

def Experiment1(X,beta,Qinv,Normal0,NormalY,sigmaY,seed):
    """ Perform the experiment described by algorithm 1 in the paper.
    Feature vectors X are fixed.
    Normal0 corresponds to the p.d.f. of model parameters under hypothesis T=0.
    NormalY corresponds to the p.d.f. of Scalar responses.
    Qinv = la.inv(np.transpose(X)@X).
    sigmaY is the std of the scalar responses.
    Qinv, Normal0 and NormalY are pre-computed for efficiency.
    """
    rd.seed(seed)
    n = X.shape[0]
    
    #Draw target attribute T
    T = rd.randint(2)
    
    #Draw index i
    j = rd.randint(n)
    
    #Draw Training Samples
    Y = DrawY(X,beta,sigmaY)
    
    #Draw Test Sample
    if T:
        S = Y[j]
    else:
        S = X[j,:]@beta + rd.normal(0,sigmaY)
    
    #Train
    betaHat = TrainLinear(X,Y,Qinv)
    
    #Training Set Error
    mse = MSE(X@betaHat,Y)
    
    #Generalization Error Estimation
    Ytest = DrawY(X,beta,sigmaY)
    genErrEmp = MSE(X@betaHat,Ytest)
    

    prob0List = []
    prob1List = []
    for k in range(n):  
        Normal1 = SamplerTheta(1,beta,Qinv,sigmaY,X,i=k,S=S)
        
        # Computing Posterior given T=0
        prob0List.append(Normal0.pdf(betaHat)*NormalY.pdf(S,k))
        
        # Computing Posterior given T=1
        prob1List.append(Normal1.pdf(betaHat)*NormalY.pdf(S,k))     
        
    prob0 = np.mean(prob0List)
    prob1 = np.mean(prob1List)
    
    Suc = int(int(prob1>prob0)==T)
    
    return Suc, genErrEmp, mse