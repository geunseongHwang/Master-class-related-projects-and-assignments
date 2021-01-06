# -*- coding: utf-8 -*-

# DO NOT CHANGE
import pandas as pd
import numpy as np

# 순서정하기
def get_order(structure):
    # structure: dictionary of structure
    #            key is variable and value is parents of a variable
    # return list of learning order of variables 
    # ex) ['A', 'R', 'E', 'S', 'T', 'O']
    
    return list(structure.keys())

# 정해진 순서에 따라 확률값 구하기
def learn_parms(data,structure,var_order):
    # data: training data
    # structure: dictionary of structure 
    # var_order: list of learning order of variables 
    # return dictionary of trained parameters (key=variable, value=learned parameters)
    
    value_1 = {}
    value_2 = {}
    value_3 = {}

    for i in range(len(structure)):

        for j in range(len(order1)):

            if list(structure.keys())[i] == var_order[j]:

                if len(list(structure.values())[i]) == 0:

                    value1 = data[list(structure.keys())[i]].value_counts()
                    value_1[order1[j]] = value1 / value1.sum()

                elif len(list(structure.values())[i]) == 1:

                    value2 = data[list(structure.keys())[i]].groupby(data[list(structure.values())[i][0]]).value_counts().unstack()

                    for i in range(len(value2)):

                        value2.iloc[i,:] = value2.iloc[i,:] / np.sum(value2.iloc[i,:])
                        value_2[order1[j]] = value2

                elif len(list(structure.values())[i]) == 2:

                    value3 = data[list(structure.keys())[i]].groupby([data[list(structure.values())[i][0]],data[list(structure.values())[i][1]]]).value_counts().unstack()

                    for i in range(len(value3)):

                        value3.iloc[i,:] = value3.iloc[i,:] / np.sum(value3.iloc[i,:])
                        value_3[order1[j]] = value3
                      
                
    return value_1, value_2, value_3


# 그림 그리기
def print_parms(var_order,parms):
    # var_order: list of learning order of variables
    # parms: dictionary of trained parameters (key=variable, value=learned parameters)
    # print the trained parameters for each variable
    for var in var_order:
        print('-------------------------')
        print('Variable Name=%s'%(var))
        #TODO: print the trained paramters
        
        for j in range(len(parms)):

            for k in range(len(parms[j])):

                if list(parms[j].keys())[k] == var:

                    print(list(parms[j].values())[k])
    
data=pd.read_csv('https://drive.google.com/uc?export=download&id=1taoE9WlUUN4IbzDzHv7mxk_xSj07f-Zt', sep=' ')

##########################################################################


str1={'A':[],'S':[],'E':['A','S'],'O':['E'],'R':['E'],'T':['O','R']}

# 순서 및 구조가 들어간 형태
order1=get_order(str1)
# dictionary 형태로 출력
parms1=learn_parms(data,str1,get_order(str1))
print('-----First Structure------')
# get_order, learn_parm이 동시에 드감 
print_parms(order1,parms1)
print('')

str2={'A':['E'],'S':['A','E'],'E':['O','R'],'O':['R','T'],'R':['T'],'T':[]}
# -> ['T', 'R', 'O', 'E','A','S']
# 순서 및 구조가 들어간 형태
order2=get_order(str2)
# dictionary 형태로 출력
parms2=learn_parms(data,str2,get_order(str2))
print('-----First Structure------')
# get_order, learn_parm이 동시에 드감 
print_parms(order2,parms2)
print('')

################ 실험용 #######################
order2=['T', 'R', 'O', 'E','A','S']
parms2=learn_parms(data,str2,['T', 'R', 'O', 'E','A','S'])
print('-----Second Structure-----')
print_parms(order2,parms2)
print('')













