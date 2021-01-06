import pandas as pd
import time
from bs4 import BeautifulSoup
import requests
import re
import json
from calendar import monthrange

class crawler_for_naver_news(object):

    def __init__(self, start_year, start_month, keyword):
        self.special_symbol = re.compile('[\{\}\[\]\/?,;:|\)*~`!^\-_+<>\#$&▲▶◆◀■【】\\\=\(\'\"]')
        self.content_pattern = re.compile('본문 내용|TV플레이어| 동영상 뉴스|flash 오류를 우회하기 위한 함수 추가function  flash removeCallback|tt|앵커 멘트|xa0')
        self.at= re.compile('@')
        self.waiting_time= 0.1
        self.base_url= "https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=105&date="
        self.start_date= start_year+ start_month+ '01'
        self.periods= monthrange(int(start_year), int(start_month))[1]
        self.date_list= [d.strftime('%Y%m%d') for d in pd.date_range(str(self.start_date), periods= self.periods)]
        self.keyword= keyword

    def get_finalpage(self, url):
        try:
            totalpage_url= url
            request_content= requests.get(totalpage_url)
            document_content= BeautifulSoup(request_content.content, 'html.parser')
            headline_tag= document_content.find('div', {'class': 'paging'}).find('strong')
            regex= re.compile(r'<strong>(?P<num>\d+)')
            match= regex.findall(str(headline_tag))
            return int(match[0])
        except Exception:
            return 0

    def get_newslist_url_list(self):
        newslist_url_list= []
        for date in self.date_list:
            url= self.base_url+ date
            finalpage= self.get_finalpage(url+ "&page=10000")
            for page in range(1, finalpage+ 1):
                newslist_url_list.append(url+ '&page='+ str(page))
        return newslist_url_list

    def get_news_url_list(self, newslist_url):
        request= requests.get(newslist_url)
        html= BeautifulSoup(request.content, 'html.parser')
        news_url_list_html= html.select('.newsflash_body .type06_headline li dl')
        news_url_list_html.extend(html.select('.newsflash_body .type06 li dl'))
        news_url_list= []
        for html in news_url_list_html:
            url= html.a.get('href')
            news_url_list.append(url)
        return news_url_list

    def get_email_address(self, article_tag):
        regex= re.compile('^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
        email= None
        for line in article_tag[0].find_all(text= True):
            line= re.sub(self.special_symbol, ' ', line)
            line= re.sub(self.content_pattern, ' ', line)
            str_list= line.split()
            for element in str_list:
                if regex.match(element)!= None:
                    email= element
        return email

    def get_article(self, article_tag):
        text= str(article_tag[0].find_all(text= True))
        text= text.replace('\\n', '').replace('\\t', '')
        text= re.sub(self.special_symbol, ' ', text)
        text= re.sub(self.content_pattern, ' ', text)
        text= re.sub(self.at, ' ', text)
        text= re.sub(' +', ' ', text).lstrip()
        reversed_text= ''.join(reversed(text))
        article= ''
        for i in range(0, len(text)):
            if reversed_text[i: i+ 2]== '.다':
                article= ''.join(reversed(reversed_text[i: ]))
                break
        return article

    def get_headline(self, headline_tag):
        headline= str(headline_tag[0].find_all(text= True))
        headline= re.sub(self.special_symbol, '', headline)
        return headline

    def get_press(self, press_tag):
        press= str(press_tag[0].get('content'))
        return press

    def get_likeit(self, news_url):
        likeit_dict= {'like': 0, 'warm': 0, 'sad': 0, 'angry': 0, 'want': 0}
        regex_oid= re.compile('oid=(\d+)')
        regex_aid= re.compile('aid=(\d+)')
        oid= regex_oid.findall(news_url)[0]
        aid= regex_aid.findall(news_url)[0]
        request= oid+ '_'+ aid
        reaction_request_url= 'https://news.like.naver.com/v1/search/contents?suppress_response_codes=true&callback=jQuery1124033427653310966177_1559044448097&q=NEWS%5Bne_'+ request+ '%5D%7CNEWS_SUMMARY%5B'+ request+ '%5D%7CNEWS_MAIN%5Bne_'+ request+ '%5D&isDuplication=false&_=1559044448098'
        reaction_request= requests.get(reaction_request_url)
        js= BeautifulSoup(reaction_request.content, 'lxml').text
        start= js.find('{')
        js_list= json.loads(js[start: -2])['contents'][0]['reactions']
        for js_state in js_list:
            likeit_dict[js_state['reactionType']]= js_state['count']
        return likeit_dict

    def get_status(self, news_url):
        regex_oid= re.compile('oid=(\d+)')
        regex_aid= re.compile('aid=(\d+)')
        oid= regex_oid.findall(news_url)[0]
        aid= regex_aid.findall(news_url)[0]
        request= oid+ '_'+ aid
        reaction_request_url= 'https://news.like.naver.com/v1/search/contents?suppress_response_codes=true&callback=jQuery1124033427653310966177_1559044448097&q=NEWS%5Bne_'+ request+ '%5D%7CNEWS_SUMMARY%5B'+ request+ '%5D%7CNEWS_MAIN%5Bne_'+ request+ '%5D&isDuplication=false&_=1559044448098'
        reaction_request= requests.get(reaction_request_url)
        return reaction_request.status_code

    def search_keyword(self, article):
        for word in article.split():
            for key in keyword:
                if key in word:
                    return True
        return False

    def crawling(self):
        newslist_url_list= self.get_newslist_url_list()
        url_list= []
        article_list= []
        date_list= []
        press_list= []
        headline_list= []
        email_list= []
        like_count_list= []
        warm_count_list= []
        sad_count_list= []
        angry_count_list= []
        want_count_list= []
        for newslist_url in newslist_url_list:
            regex= re.compile('date=(\d+)')
            newsdate= regex.findall(newslist_url)[0]
            news_url_list= self.get_news_url_list(newslist_url)
            time.sleep(self.waiting_time)
            for news_url in news_url_list:
                print(news_url+ ' | '+ newsdate+ ' | '+ str(self.get_status(news_url)))
                request_content= requests.get(news_url)
                time.sleep(self.waiting_time)
                soup= BeautifulSoup(request_content.content, 'html.parser')
                headline_tag= soup.find_all('h3', {'id': 'articleTitle'}, {'class': 'tts_head'})
                article_tag= soup.find_all('div', {'id': 'articleBodyContents'})
                press_tag= soup.find_all('meta', {'property': 'me2:category1'})
                #예외처리
                if not article_tag:
                    continue
                article= self.get_article(article_tag)
                #기사에 키워드가 존재하지 않으면 수집하지 않음
                if not self.search_keyword(article):
                    continue
                headline= self.get_headline(headline_tag)
                email= self.get_email_address(article_tag)
                press= self.get_press(press_tag)
                likeit_dict= self.get_likeit(news_url)
                #각 요소들을 list로 저장
                url_list.append(news_url)
                article_list.append(article)
                date_list.append(newsdate)
                press_list.append(press)
                headline_list.append(headline)
                email_list.append(email)
                like_count_list.append(likeit_dict['like'])
                warm_count_list.append(likeit_dict['warm'])
                sad_count_list.append(likeit_dict['sad'])
                angry_count_list.append(likeit_dict['angry'])
                want_count_list.append(likeit_dict['want'])
        df= {}
        df['headline']= headline_list
        df['date']= date_list
        df['article']= article_list
        df['press']= press_list
        df['like']= like_count_list
        df['warm']= warm_count_list
        df['sad']= sad_count_list
        df['angry']= angry_count_list
        df['want']= want_count_list
        df['email']= email_list
        df['url']= url_list
        df= pd.DataFrame(df, columns= ['headline', 'date', 'article', 'press', 'like', 'warm', 'sad', 'angry', 'want', 'email', 'url'])
        return df
