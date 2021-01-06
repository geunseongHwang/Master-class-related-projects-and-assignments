import pandas as pd
import numpy as np
import csv
import glob
import os

# 경로 변경 #
os.chdir(r'C:\Users\User\Desktop\뉴스data2017')

# 현재 경로 #
print(os.getcwd())

class overall_data_processing(object):
    
    def __init__(self, input_path, input_file_name, output_file_name,file_name,form,
                 Words_to_remove,select_columns,sample_n, save):
        self.input_path = input_path
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.file_name = file_name
        self.form = form
        self.Words_to_remove = list(Words_to_remove)
        self.select_columns = list(select_columns)
        self.sample_n = sample_n
        self.save = bool(save)

    def month_to_half_year(self):
        '''
        해당되는 파일 전부 읽어들여 반기단위로 csv파일 처리
        출처: https://formyproj.tistory.com/34
        '''
        is_first_file = True
    
        # 규칙에 따라 지정된 패턴과 일치하는 모든 경로명을 찾음 #
        for input_file in glob.glob(os.path.join(self.input_path, '%s*') %self.input_file_name):
        
            # 입력받은 경로의 기본 이름(base name)을 반환 #
            print(os.path.basename(input_file))
            
            # .csv 파일 오픈 #
            with open(input_file, 'r', newline='', encoding='UTF8') as csv_in_file:
                
                # .csv 파일 추가 오픈 #
                with open(self.output_file_name, 'a', newline='' , encoding='UTF8') as csv_out_file:
                    
                    # 파일 객체 읽기 #
                    freader = csv.reader(csv_in_file)
                    
                    # 파일 객체 쓰기 #
                    fwriter = csv.writer(csv_out_file)
                    
                    # is_first_file로 첫번째 header 행만 추가하기 위해 나눔 #
                    if is_first_file:
                        for row in freader:
                            
                            # data list를 라인마다 추가 #
                            fwriter.writerow(row)
                            
                        is_first_file = False
                        
                    else:
                        # columns name이 추가되는 것을 방지 #
                        header = next(freader)
                        
                        for row in freader:
                            fwriter.writerow(row)
    
    def load_file_csv(self) :
        '''
        원하는 반기 데이터 불러오기 
        '''
        
        self.month_to_half_year()
        
        # 폴더에 속해있는 모든 파일 리스트로 불러오기 #
        file_list = os.listdir(self.input_path)
        
        # 모든 파일 중 csv파일만 불러오기 #
        file_list_csv = [file for file in file_list if file.endswith(".csv")]
        print ("file_list_csv: {}".format(file_list_csv))
        
        # csv파일 중 원하는 반기 데이터 파일만 가져오기 #
        for select_csv_file in file_list_csv:
           if 'naver_news_%s'%file_name in select_csv_file:
               wanted_half_data_file = pd.read_csv(select_csv_file)
                                              
        return wanted_half_data_file
    
    def drop_dupliecate(self) :
        '''
        중복된 기사 및 결측값이 있는 행 제거
        '''
        data = self.load_file_csv()
        
        dupl_articles_elim = data.drop_duplicates([self.form], inplace=False)
        mis_val_elim = dupl_articles_elim.dropna(how='any', axis=0, inplace=False)
        
        return mis_val_elim
    
    def drop_none_interest(self) :
        '''
        감정 표현이 없는 기사 제거
        '''
        data = self.drop_dupliecate() 
        
        zero_emotion = data.loc[data.like == 0].loc[data.warm == 0].loc[data.sad == 0].loc[data.angry == 0]
        emotion_removal = data[np.in1d(data.index, zero_emotion.index) == False]
        
        return emotion_removal
        
    def reactions(self):
        '''
        긍정반응 < 부정반응 기사제거
        '''
        
        data = self.drop_none_interest()
        
        reaction = data.loc[(data.like + data.warm + data.want)*3 < data.sad + data.angry]
        reaction = data[np.in1d(data.index, reaction.index) == False]
        
        return reaction
    
    def drop_adv(self) :
        '''
        제목 광고성 기사 제거
        '''
        data = self.reactions()
        
        contains = []
        
        for i in self.Words_to_remove:
            contains.append(data['headline'].loc[data['headline'].str.contains(i)])
            
        for j in contains:
            Remove_Ads = data[np.in1d(data.index, j.index) == False]
            
        return Remove_Ads
    
    def article_content_processing(self):
        '''
        기사 내용이 100자 미만인 경우 제거
        '''
        
        data = self.drop_adv()
        
        hundred_elim = data.iloc[np.array([len(i) for i in data['article']]) < 100]
        article_pro = data[np.in1d(data.index, hundred_elim.index) == False]
        
        return article_pro
        
    def random_sampling(self, final_csv_name):
        '''
        날짜별 원하는 컬럼 선택 후 random_sampling
        '''
        
        data = self.article_content_processing()
        
        data = data[self.select_columns].sort_values(by='date')
        
        data = data.sample(n=self.sample_n)
    
        if self.save == True:
            data.to_csv('%s.csv'%final_csv_name,index=False, encoding='utf-8' )
            
        elif self.save == False:
            return data
            
        else:
            print("Error!!")
            
########
# test #
########
    
# test # 
in_path = './'
in_name = 'naver_news_2017_first'
out_name = 'naver_news_2017_half_first.csv'
file_name = '2017_half_first'
form = 'article'
Words_to_remove = '출시', '역사', '예약', '문학', '광고', '지원금', \
                   '게임', '출시', '예정', '국제전화', '공개', '출시'
select_columns = 'headline','date','article','press'
sample_n = 500
save = True
final_csv_name = 'news_data_2017_the_first_half(500)'

Over_data_pro = overall_data_processing(in_path, in_name, out_name,file_name,form,
                             Words_to_remove,select_columns,sample_n, save)

Over_data_pro.random_sampling(final_csv_name)

test_file = pd.read_csv('news_data_2017_the_first_half(500).csv')




