# -*- coding: utf-8 -*-

# DO NOT CHANGE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


def create_bootstrap(X,y,ratio):
    # X: input data matrix
    # ratio: sampling ratio
    # return bootstraped dataset (newX,newy)
    
    dataset = np.random.choice(len(X), replace=True, size=int(len(X)*ratio))
    
    new_X = []
    new_y = []
    
    for i in dataset:
        if i == np.arange(len(X))[i]:
            new_X.append(list(X[i]))
            new_y.append(y[i])

    return np.array([new_X, new_y])

## bagging으로부터 뽑힌 knn의 예측값을 보고 다수인 것을 class로 뱉음
## 만약 클라스가 2, 3개로 다 같으면 맨 앞에 클라스로 예측함

def voting(y):
    # y: 2D matrix with n samples by n_estimators
    # return voting results by majority voting (1D array)
    
    y = y.astype('int')
    
    voting_set = []
    
    for i in range(len(y)):
        max_freq_num = np.bincount(y[i], minlength=3) == np.bincount(y[i]).max()
        voting_set.append(y[i][max_freq_num][0])
        
    
    return np.array(voting_set)
    
# bagging
def bagging_cls(X,y,n_estimators,k,ratio):
    # X: input data matrix
    # y: output target
    # n_estimators: the number of classifiers
    # k: the number of nearest neighbors
    # ratio: sampling ratio
    # return list of n k-nn models trained by different boostraped sets
    
    boostraped_sets = []

    while n_estimators != 0:
        
        knn=KNeighborsClassifier(n_neighbors=k)
        dataset = np.random.choice(len(X), replace=True, size=int(len(X)*ratio))
    
        X_select = []
        y_select = []
        
        for i in dataset:
            if i == np.arange(len(X))[i]:
                X_select.append(X[i])
                y_select.append(y[i])
        
        boostraped_sets.append(knn.fit(X_select,y_select))
        
        n_estimators -= 1
        
    
    return boostraped_sets
    
    
    
data=load_iris()
X=data.data[:,:2]
y=data.target    

n_estimators=3 # knn model 3개
k=3
ratio=0.8

# sepal length, sepal width의 column을 기준으로 min max의 범위 정함
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# 정해진 범위 사이로 값을 0.02단위로 세분화
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# new data? 2D column(N X 2)으로 만듦
Z = np.c_[xx.ravel(), yy.ravel()]

##
models = bagging_cls(X,y,n_estimators,k,ratio)
y_models = np.zeros((len(xx.ravel()), n_estimators))
for i in range(n_estimators):
    y_models[:,i]=models[i].predict(Z)

y_pred=voting(y_models)


# Draw decision boundary
plt.contourf(xx,yy, y_pred.reshape(xx.shape), cmap=plt.cm.RdYlBu)
plt.scatter(X[:,0],X[:,1], c='k', s=10)

len(create_bootstrap(X,y,ratio)[0])
len(X)

