# -*- coding: utf-8 -*-
 
# DO NOT CHANGE
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

def create_bootstrap(X,y,ratio):
    # X: input data matrix
    # ratio: sampling ratio
    # return one bootstraped dataset and indices of sub-sampled samples (newX,newy,ind)
    
    # indices: out of sample
    # ratio가 100%이므로 569개 data가 다들어가는 것이 맞지만 이 569개는 중복포함
    
    dataset = np.random.choice(len(X), replace=True, size=int(len(X)*ratio))
            
    return [dataset, np.setdiff1d(np.arange(len(X)), np.unique(dataset))]

def cal_oob_error(X,y,models,ind):
    # X: input data matrix
    # y: output target
    # models: list of trained models by different bootstraped sets
    # ind: list of indices of samples in different bootstraped sets
    
    test_dt = []

    for i in range(len(models)):
        i_d = np.setdiff1d(np.arange(len(X)), np.unique(ind[i]))
        
        
        Xx = np.zeros(len(X), dtype=float)
        Xx.fill(np.nan)
        
    
        Xx[i_d] = y[i_d] == models[i].predict(X)[i_d]
    
        test_dt.append(Xx) # 569 X 500
    
    OOB_value_Error = []
    
    for j in range(len(np.c_[test_dt].T)):
        OOB_value_Error.append(np.sum(np.c_[test_dt].T[j] == 0) \
                               / (np.sum(np.c_[test_dt].T[j] == 0) + np.sum(np.c_[test_dt].T[j] == 1)))

    
    return OOB_value_Error
    

def cal_var_importance(X,y,models,ind,oob_errors):
    # X: input data matrix
    # y: output target
    # models: list of trained models by different bootstraped sets
    # ind: list of indices of samples in different bootstraped sets
    # oob_errors: list of oob error of each sample
    # return variable importance
    
    new_OOB = []
    
    for j in range(len(X[0,:])):
    
        feature_shuffle = shuffle(X[:,j])
    
        Existing_X = np.delete(X, j, axis=1) 
    
        new_S_X = np.insert(Existing_X, j, feature_shuffle, axis=1)
       
        new_OOB.append(new_S_X)
    
    variable_importance = []
    
    for i in range(len(new_OOB)):
        
        OOB_feature_Error = []
        test_dt = []
        
        for k in range(len(ind)):
            
            i_d = np.setdiff1d(np.arange(len(new_OOB[i])), np.unique(ind[k]))
            
            Xx = np.zeros(len(new_OOB[i]), dtype=float)
            Xx.fill(np.nan)
            
            Xx[i_d] = y[i_d] == models[k].predict(new_OOB[i])[i_d]
            
            test_dt.append(Xx) # 569 X 500
            
        OOB_feature_Error_raw = []
        
        for j in range(len(np.c_[test_dt].T)):
            OOB_feature_Error_raw.append(np.sum(np.c_[test_dt].T[j] == 0) \
                                       / (np.sum(np.c_[test_dt].T[j] == 0) + np.sum(np.c_[test_dt].T[j] == 1)))
            
        for l in range(len(OOB_feature_Error_raw)):
            error_difference = OOB_feature_Error_raw[l]  - oob_errors[l]
            
            OOB_feature_Error.append(error_difference)
            
        variable_importance.append(np.sum(OOB_feature_Error)/len(OOB_feature_Error))
        
   
    return variable_importance


def random_forest(X,y,n_estimators,ratio,params):
    # X: input data matrix
    # y: output target
    # n_estimators: the number of classifiers
    # ratio: sampling ratio for bootstraping
    # params: parameter setting for decision tree
    # return list of tree models trained by different bootstraped sets and list of indices of samples in different bootstraped sets
    # (models,ind_set)
    
    train_random_forest = []
    new_SET = []
    
    while n_estimators != 0:
    
        DTC = DecisionTreeClassifier(max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'])
        
        dataset, ind = create_bootstrap(X,y,ratio)
        
        newX, newy = X[dataset], y[dataset] 
        
        new_SET.append(dataset)
        
        train_random_forest.append(DTC.fit(newX,newy))

        n_estimators -= 1
        
    return list(train_random_forest), list(new_SET)
    
data=datasets.load_breast_cancer()
X, y = shuffle(data.data, data.target, random_state=13)

params = {'max_depth': 4, 'min_samples_split': 0.1, 'min_samples_leaf':0.05}
n_estimators=500
ratio=1.0

models, ind_set = random_forest(X,y,n_estimators,ratio,params)
oob_errors=cal_oob_error(X,y,models,ind_set)
var_imp=cal_var_importance(X,y,models,ind_set,oob_errors)

nfeature=len(X[0])
plt.barh(np.arange(nfeature),var_imp/sum(var_imp))
plt.yticks(np.arange(nfeature) + 0.35 / 2, data.feature_names)


#############################################





