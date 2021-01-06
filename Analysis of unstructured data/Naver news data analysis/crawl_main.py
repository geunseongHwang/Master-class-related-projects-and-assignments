import pandas as pd
import time
from bs4 import BeautifulSoup
import requests
import re
import json
from calendar import monthrange
import argparse
from crawler import crawler_for_naver_news
import os

paser = argparse.ArgumentPaser()

parser.add_argument('--keyword', default='인공지능', nargs='+', type=str)
parser.add_argument('--year', default=['2019'], nargs='+', type=str)
parser.add_argument('--start-month', default=1, type=int, help="If you want selected month")
parser.add_argument('--end-month', default=13, type=int, help="If you want selected month")
parser.add_argument('--exp', default=test, type=str)

args = parser.parse_args()

be = "keep"

if not os.path.exists(args.exp) :
    os.makedirs(args.exp)

for y in args.year:
    for m in range(args.start_month, args.end_month):
        if y == '2019' and m == 12 :
            break
        m= str(m).zfill(2)
        crawler= crawler_for_naver_news(start_year= y, start_month= m, keyword= args.keyword)
        df= crawler.crawling()
        df.to_csv(path_or_buf= args.exp + y+ m+ '.csv', sep= ',', encoding= 'utf-8', index= False)
