# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:09:13 2020

@author: User
"""

import numpy as np
import pandas as pd 


from sklearn.model_selection import train_test_split  

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import cross_validate 
import matplotlib.pyplot as plt
import seaborn as sns

from sentiment_sentence_processing import sentence_processing

neg_path = r"C:\Users\User\Desktop\대학원 과제모음\1학년 2학기\심화기계\머신러닝 팀플\4조_논문구현\rt-polaritydata\rt-polarity.neg.txt"
pos_path = r"C:\Users\User\Desktop\대학원 과제모음\1학년 2학기\심화기계\머신러닝 팀플\4조_논문구현\rt-polaritydata\rt-polarity.pos.txt"

# 부모노드 #
class Overall_Data_split():
    
    def __init__(self, neg_path, pos_path):
    
        self.overall_data =  sentence_processing(neg_path, pos_path).sentiment_raw_data()
        
        
    def data_split(self):
        text_data = [self.overall_data[i][0] for i in range(len(self.overall_data))]
        label = [self.overall_data[i][1] for i in range(len(self.overall_data))]
    
        X_train, X_test, y_train, y_test = train_test_split(text_data, label, test_size=0.2, random_state=42)
        
        data_splits = {'X_train':X_train, 'X_test': X_test, \
                       'y_train':y_train, 'y_test': y_test}
        
        return data_splits
    
# 자식노드들 #
class Weighting_Scheme(Overall_Data_split):

    def __init__(self, scheme):
    
        self.sheme = scheme
        
    def Scheme_select(self):
        
        # pipeline에 weighting scheme와 model 선언 #
        # CRF대신 RigdeClassifier 모델을 사용
        
        X_train = Overall_Data_split(neg_path,pos_path).data_split()['X_train']
        #X_test = Overall_Data_split(neg_path,pos_path).data_split()['X_test']
        y_train = Overall_Data_split(neg_path,pos_path).data_split()['y_train']
        #y_test = Overall_Data_split(neg_path,pos_path).data_split()['y_test']
        
        if self.sheme == 'BL':
            
            ###########
            # Boolean #
            ###########
            
            SGDClassifier_Pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True),), 
                                       ('clf', SGDClassifier(loss='log'))],)
    
            LogisticRegression_Pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True)),
                             ('clf', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)),])
            
            MultinomialNB_Pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True),),
                             ('clf', MultinomialNB()),])
        
            RidgeClassifier_Pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), lowercase=True)),
                             ('clf', CalibratedClassifierCV(RidgeClassifier(solver='sparse_cg'), cv=5)),])
        
        elif self.sheme == 'TF':
            
            ######
            # TF #
            ######
    
            SGDClassifier_Pipeline = Pipeline([
                     ('tf', TfidfVectorizer(use_idf=False,ngram_range=(1, 2))),
                     ('clf', SGDClassifier(loss='log'))],)
    
            LogisticRegression_Pipeline = Pipeline([
                             ('tf', TfidfVectorizer(use_idf=False,ngram_range=(1, 2))),
                             ('clf', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)),])
            
            MultinomialNB_Pipeline = Pipeline([
                             ('tf', TfidfVectorizer(use_idf=False,ngram_range=(1, 2))),
                             ('clf', MultinomialNB()),])
            
            RidgeClassifier_Pipeline = Pipeline([
                             ('tf', TfidfVectorizer(use_idf=False,ngram_range=(1, 2))),
                             ('clf', CalibratedClassifierCV(RidgeClassifier(solver='sparse_cg'), cv=5)),])
    
        elif self.sheme == 'TF-IDF':
            
            ##########
            # TF-IDF #
            ##########
                
            SGDClassifier_Pipeline = Pipeline([
                     ('tfidf', TfidfVectorizer(use_idf=True,ngram_range=(1, 2))),
                     ('clf', SGDClassifier(loss='log'))],)
    
            LogisticRegression_Pipeline = Pipeline([
                             ('tfidf', TfidfVectorizer(use_idf=True,ngram_range=(1, 2))),
                             ('clf', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)),])
            
            MultinomialNB_Pipeline = Pipeline([
                             ('tfidf', TfidfVectorizer(use_idf=True,ngram_range=(1, 2))),
                             ('clf', MultinomialNB()),])
            
            RidgeClassifier_Pipeline = Pipeline([
                             ('tfidf', TfidfVectorizer(use_idf=True,ngram_range=(1, 2))),
                             ('clf', CalibratedClassifierCV(RidgeClassifier(solver='sparse_cg'), cv=5)),])
        else:
            print("올바르지 않은 입력입니다.")
            
        # 어떤 metric을 가져올건지 선언 #
        scoring = ['precision_weighted','recall_weighted','f1_weighted','accuracy']
        
        SGD_Score = cross_validate(SGDClassifier_Pipeline, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
        LR_Score = cross_validate(LogisticRegression_Pipeline, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
        MNB_Score = cross_validate(MultinomialNB_Pipeline, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
        RC_Score = cross_validate(RidgeClassifier_Pipeline, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    
        # model Metric #
        recall_raw_score = SGD_Score['test_recall_weighted'].mean() + LR_Score['test_recall_weighted'].mean() \
                             + MNB_Score['test_recall_weighted'].mean() + RC_Score['test_recall_weighted'].mean()
        
        precision_raw_score = SGD_Score['test_precision_weighted'].mean() + LR_Score['test_precision_weighted'].mean() \
                             + MNB_Score['test_precision_weighted'].mean() + RC_Score['test_precision_weighted'].mean()
        
        f1_raw_score = SGD_Score['test_f1_weighted'].mean() + LR_Score['test_f1_weighted'].mean() \
                             + MNB_Score['test_f1_weighted'].mean() + RC_Score['test_f1_weighted'].mean()
       
        accuracy_raw_score = SGD_Score['test_accuracy'].mean() + LR_Score['test_accuracy'].mean() \
                             + MNB_Score['test_accuracy'].mean() + RC_Score['test_accuracy'].mean()
    
    
        total_recall = recall_raw_score / len(scoring)
        total_precision = precision_raw_score / len(scoring)
        total_f1 = f1_raw_score / len(scoring)
        total_accuracy = accuracy_raw_score / len(scoring)
    
        total_metrics = {'recall':total_recall, 'precision': total_precision, \
               'f1':total_f1, 'accuracy': total_accuracy}
        
        return total_metrics
    
# test용
Overall_Data_split(neg_path,pos_path).data_split()

WS =  Weighting_Scheme('ghf')

WS.Scheme_select()
    
    
    