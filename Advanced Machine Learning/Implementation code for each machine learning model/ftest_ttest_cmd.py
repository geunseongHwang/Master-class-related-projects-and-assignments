# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 19:59:38 2018

@author: Administrator
"""

# Use only following packages
import numpy as np
from scipy import stats
from sklearn.datasets import load_boston

def ftest(X,y):
    # X: inpute variables
    # y: target
    
    n,p = len(X), X.shape[1] 

    X=np.c_[np.ones(n), X]

    xtx=np.matmul(X.T, X)
    xtx_inv=np.linalg.inv(xtx)

    beta=np.matmul(np.matmul(xtx_inv, X.T), y)
    
    y_pred=np.matmul(X,beta)
    
    SSE = np.round_(sum((y-y_pred)**2), 4)
    SST = np.round_(sum((y-y.mean())**2), 4)
    SSR = np.round_(sum((y_pred-y.mean())**2), 4)
    
    f = 0
    pvalue = 0
    
    MSE=np.round_(SSE/(n-p-1), 4)
    MSR=np.round_(SSR/p, 4)
    
    f = np.round_(MSR/MSE, 4)

    pvalue = 1-stats.f.cdf(f,p,n-p-1)
    pvalue = "{:.4f}".format(pvalue)
    
    One_ANOVA = [["Factor", "SS", "DF","MS","F-value","Pr>F"], \
                 ["Model", SSR, p, MSR, f, pvalue], \
                 ["Error", SSE, n-p-1, MSE], \
                 ["Total", SST, n-1]]

    print("---------------------------------------------------------------------- ")
    for line in One_ANOVA:
        if len(line) == 6:
            print("{:^8} {:>12} {:>8} {:>12} {:>12} {:>12}".format(*line))
        
        elif len(line) == 4:
            print("{:^8} {:>12} {:>8} {:>12} ".format(*line))
    
        else:
            print("--------------------------------- ")
            print("{:^8} {:>12} {:>8}".format(*line))
    
    
    return 0

def ttest(X,y, varname):
    # X: inpute variables
    # y: target
    
    n,p = len(X), X.shape[1]
    
    X = np.c_[np.ones(n), X]
    
    xtx = np.matmul(X.T, X)
    xtx_inv=np.linalg.inv(xtx)
    
    beta= np.matmul(np.matmul(xtx_inv, X.T), y)
    
    y_pred=np.matmul(X,beta)
    
    SSE = np.round_(sum((y-y_pred)**2), 4)
    
    MSE=np.round_(SSE/(n-p-1), 4)
    
    t = []
    for i in range(p+1):
        t_raw = np.round_((beta[i])/np.sqrt(MSE*xtx_inv[i,i]),4)
        t.append(t_raw)
    
    se = []
    for i in range(len(beta)):
        se_raw = np.sqrt(MSE * xtx_inv[i,i])
        se_raw = np.round_(se_raw, 4)
        se.append(se_raw)
    
    pvalue=[]
    for i in range(p+1):
        pvalue.append((1-stats.t.cdf(np.abs(t[i]),n-p-1))*2)
        
    pvalue = ["{:.4f}".format(pvalue[i]) for i in range(len(pvalue))]
    beta= ["{:.4f}".format(beta[i]) for i in range(len(beta))]
    
    varname = np.array(varname, dtype=np.string_)
    varname = list(varname)
    varname = [varname[i].decode() for i in range(len(varname))]
    
    summery = [["Variable", "coef", "se", "t", "Pr>F"]]
    
    for i in range(len(beta)):
        
        if beta[i] == beta[0]: 
            summery.append(["Const", beta[i], se[i], t[i], pvalue[i]])
            
        else:
            summery.append([varname[i-1], beta[i], se[i], t[i], pvalue[i]])
    
    
    print("------------------------------------------------------------- ")
    for line in summery:
         print("{:^8} {:>12} {:>12} {:>12} {:>12}".format(*line))
        
    print("------------------------------------------------------------- ")         
    
    return 0

## Do not change!
# load data
data=load_boston()
X=data.data
y=data.target

ftest(X,y)
ttest(X,y, varname=data.feature_names)
















