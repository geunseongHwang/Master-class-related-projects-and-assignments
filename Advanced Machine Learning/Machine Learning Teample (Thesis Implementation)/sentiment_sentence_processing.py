# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 00:50:13 2020

@author: User
"""
import pandas as pd
import re

##############
# processing #
##############

# 부정 및 긍정 문장 전처리 과정 #
class sentence_processing():
    
    def __init__(self, n_path, p_path):
        # 경로 설정 #
        self.n_path = n_path
        self.p_path = p_path
    
    def sentiment_raw_data(self):
        document = []
        
        # 텍스트 파일 불러오기 #
        with open(self.n_path, "r") as f:
            for neg_sentence in f:
                document.append((neg_sentence, 0))
        
        with open(self.p_path, "r") as f:
            for pos_sentence in f:
                document.append((pos_sentence, 1))
        
        document = pd.DataFrame(document)
        
        # lbael설정 #
        X = document.iloc[:,0].values
        y = document.iloc[:,1].values
    
        processed_Data_ = []
        
        # 특정문자제거 #
        for Data in range(0, len(X)):  
            # Remove all the special characters
            processed_Data = re.sub(r'\W', ' ', str(X[Data]))
         
            # remove all single characters
            processed_Data = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_Data)
         
            # Remove single characters from the start
            processed_Data = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_Data) 
         
            # Substituting multiple spaces with single space
            processed_Data= re.sub(r'\s+', ' ', processed_Data, flags=re.I)
         
            # Removing prefixed 'b'
            processed_Data = re.sub(r'^b\ss+', '', processed_Data)
            
            # Converting to Lowercase
            processed_Data = processed_Data.lower()
         
            processed_Data_.append(processed_Data)
    
        # 전처리후 부정 및 긍정 문장 통합 #
        complete_datset = [(processed_Data_[i], y[i]) for i in range(len(processed_Data_))]
            
        return complete_datset

# test #
#neg_path = r"C:\Users\User\Desktop\대학원 과제모음\1학년 2학기\심화기계\머신러닝 팀플\4조_논문구현\rt-polaritydata\rt-polarity.neg.txt"
#pos_path = r"C:\Users\User\Desktop\대학원 과제모음\1학년 2학기\심화기계\머신러닝 팀플\4조_논문구현\rt-polaritydata\rt-polarity.pos.txt"

#sp = sentence_processing(neg_path, pos_path)
#sp.sentiment_raw_data()


