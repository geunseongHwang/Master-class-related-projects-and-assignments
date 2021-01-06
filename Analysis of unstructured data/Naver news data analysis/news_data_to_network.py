import numpy as np
import pandas as pd
import re
import konlpy
from konlpy import jvm
jvm.init_jvm()
from konlpy.tag import Okt
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning, module='konlpy')
from collections import Counter
import csv
import time
import itertools
import numpy as np

def MakingEdgeNodelist(keyword_vec,sep_sign):
    # (1) Keyword list (data set for combination)
    keywords_list =[]
    for words in keyword_vec:
        #print("words :", words)
        split_words = words.split(sep_sign) # sep_sign으로 구분된 데이터를 분리하여
        split_words = [ x.replace(" ","") for x in split_words]  # 리스트로 저장
        soted_words = sorted(split_words) # ordering
        keywords_list.append(soted_words)

    # (2) Combination
    comb_list = []
    for i in keywords_list:
        for subset in itertools.combinations(i, 2):
            if subset[0] != subset[1]: # 같은 단어의 컴비네이션 제외
                comb_list.append(subset)

    # (3) Make dict & count
    comb_count = {}
    for combi in comb_list:
        comb_count[combi] = comb_count.get(combi,0)+1

    # (4) dic to df
    comb_df = pd.DataFrame()

    ## (4-1) dic_key to list
    comb_keylist = []
    for key in comb_count.keys():
        combkey_tolist = list(key)
        comb_keylist.append(combkey_tolist)

    ## (4-2) dic_key split
    source_list = []
    target_list = []
    for i in range(len(comb_keylist)):
        source = comb_keylist[i][0]
        target = comb_keylist[i][1]

        source_list.append(source)
        target_list.append(target)

    comb_df['Source_Label'] = source_list
    comb_df['Target_Label'] = target_list
    comb_df['weight'] = comb_count.values()


    comb_df = comb_df.sort_values('Source_Label',axis=0).reset_index(drop=True)

    # (5) node data
    total_tech_list = []
    for tech_li in keywords_list:
        tech = list(set(tech_li))
        total_tech_list.extend(tech)
    # node_data = pd.concat([comb_df['Source_Label'],comb_df['Target_Label']],axis=0) # 총 출현 기술 ; 잘못됨을 확인함

    node_df = pd.DataFrame()

    node_count = {}
    for node in total_tech_list:
        node_count[node] = node_count.get(node,0)+1

    node_df['label'] = node_count.keys()
    node_df['count'] = node_count.values()
    node_df['id'] = node_df.index


    # (6) make id
    node_id = node_df[['label','id']]
    node_id.columns = ['Source_Label','Source']

    merge_sid = pd.merge(comb_df,node_id,how = 'inner')

    node_id.columns = ['Target_Label','Target']

    merge_tid = pd.merge(merge_sid,node_id,how = 'inner') # merge 과정에서 순서가 바뀜

    edge_df = merge_tid.sort_values(['Source_Label','Target_Label'])

    edge_df['Type'] = ['Undirected']*len(edge_df)

    #edge_cols = ['Source','Target','weight','Type','Source_Label','Target_Label']
    #edgelist = edgelist[edge_cols]

    return(edge_df,node_df)

def Okt_pos(df,extracted_pos,min_len=2):
    article_list=df['article']
    result=[]
    for article in article_list:
        pos_list = Okt.pos(article,stem=True, norm=False)
        noun_list = [pos[0] for pos in pos_list if pos[1] in extracted_pos]
        text = [word for word in noun_list if len(word)>=min_len]
        result.append(text)
    return result

def remove_stop_words(article_noun, stop_word_list):
    result= []
    for article in article_noun:
        article_ = [word for word in article if word not in stop_word_list]
        result.append(article_)
    return result

# Load Data

documnet=pd.read_csv(r"PLZ_INPUT_YOUR_PATH",header=0,encoding = 'utf-8')

Okt = Okt()

extracted_pos =['Noun']

article_noun_unique = Okt_pos(documnet,extracted_pos, min_len=2)
word_count = Counter([word for words in article_noun_unique for word in words])
word_common = word_count.most_common()

stop_word_list= ["통해","위해","관련","기자","이번","때문","라며","대해","한편","또한","통한"]
article_noun_unique = remove_stop_words(article_noun_unique, stop_word_list)

news_list=[]
for word_list in article_noun_unique:
    news_list.append(','.join([word for word in word_list]))

data = pd.DataFrame(news_list,columns=["text"])

cpc_len = [ len(x.split(",")) for x in data.text]

data['cpc_len'] = cpc_len

dataset = data.loc[data.cpc_len>1] # , 로 구분되니 한개짜리는 떨군다.

# edge / node list
edge_list = pd.DataFrame()
node_list = pd.DataFrame()

s_time = time.time()
#타임시리즈와다르게for문이필요가없다.
doc = dataset['text']

edge_part,node_part = MakingEdgeNodelist(doc,',')

edge_list = pd.concat([edge_list,edge_part],axis=0)

node_list = pd.concat([node_list,node_part],axis=0)

e_time = time.time()
print("Execution_time :",e_time - s_time)

#
print(node_list.head())
print(edge_list.head())

edge_list=edge_list[edge_list['weight']>200]
node_list=node_list[node_list['count']>4]

node_list.to_csv("node_list_2019_sec.csv",index=False)
edge_list.to_csv("edge_list_2019_sec.csv",index=False)
