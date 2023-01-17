import sys    # 시스템
import os     # 시스템

import pandas as pd    # 판다스 : 데이터분석 라이브러리
import numpy as np     # 넘파이 : 숫자, 행렬 데이터 라이브러리

from bs4 import BeautifulSoup     # html 데이터 전처리
from selenium import webdriver    # 웹 브라우저 자동화
import time                       # 시간 지연
from tqdm import tqdm  # 진행상황 표시
from selenium.webdriver.chrome.service import Service
import json
from selenium.webdriver.common.by import By
from pymongo import MongoClient
  # Provide the mongodb atlas url to connect python to mongodb using pymongo
CONNECTION_STRING = "mongodb+srv://nlp-08:finalproject@cluster0.rhr2bl2.mongodb.net/?retryWrites=true&w=majority"
    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
client = MongoClient(CONNECTION_STRING)
dblist=client.list_database_names()
db = client.test_database
blogs = db.blogs
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')

#s=Service('./chromedriver')
chrome_options.add_argument("--disable-dev-shm-usage")

#driver = webdriver.Chrome('./chromedriver.exe',options=chrome_options)
keywords =['뉴욕','LA']#키워드 리스트 
for keyword in keywords: 
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("http://www.naver.com")
    time.sleep(2)
    element = driver.find_element("id","query") #f12눌러서 확인
    element.send_keys(keyword)
    element.submit()
    time.sleep(1)
    driver.find_element("link text","VIEW").click()
    driver.find_element("link text","블로그").click()
    time.sleep(1)
    driver.find_element("link text","옵션").click()
#    item_li = driver.find_elements("css selector",".option .txt")
#    item_li[4].click() #관련도 순 최신순으로 하든 1050개 제한이 있음
    item_li = driver.find_elements("css selector",".option .txt")
    item_li[12].click()
    def scroll_down(driver):
        driver.execute_script("window.scrollTo(0, 99999999)")
        time.sleep(1)
    n = 0 
    i = 70
    while i < n: #스크롤당 30
        scroll_down(driver)
        i = i+1
    url_list = []
    title_list = []

    # URL_raw 크롤링 시작
    article_raw = driver.find_elements("css selector",".api_txt_lines.total_tit")

    # 크롤링한 url 정제 시작
    for article in article_raw:
        url = article.get_attribute('href')   
        url_list.append(url)

    time.sleep(1)

    # 제목 크롤링 시작    
    #for article in article_raw:
    #    title = article.text
    #    title_list.append(title)
    number = len(url_list)
    blogli=[]
    for i in tqdm(range(0, 3)):
        # 글 띄우기
        url = url_list[i]
        
        # 크롤링

        try : 
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)   # 글 띄우기

            # 글의 iframe 접근
            driver.switch_to.frame("mainFrame")

            target_info = {}  # 개별 블로그 내용을 담을 딕셔너리 생성

            # 제목

            overlays = ".se-module.se-module-text.se-title-text"                        
            tit=driver.find_elements("css selector",overlays)
            title = tit[0].text

            # 닉네임
            overlays = ".nick"                                 
            nick = driver.find_elements("css selector",overlays)
            nickname = nick[0].text

            # 시간
            overlays = ".se_publishDate.pcol2"    
            date = driver.find_elements("css selector",overlays)
            datetime = date[0].text
 
            #공감(좋아요)
            overlays = ".u_cnt._count"                                 
            like = driver.find_elements("css selector",overlays)
            like=like[1].text

            #저작권
            overlays = ".wrap_ico_ccl"                                 
            copyright = driver.find_elements("css selector",overlays)

            if copyright == []:
                continue
            else:
                copyright=copyright[0].text
                copyli=copyright.split('\n')        
                target_info['copyright']=copyli

            # 내용 
            overlays = ".se-component.se-text.se-l-default"                                 
            contents = driver.find_elements("css selector",overlays)

            content_list = []
            for content in contents:
                content_list.append(content.text)
            content_str = ' '.join(content_list)
            #content_str=preprocess(content_str)
            # 크롤링한 글은 target_info라는 딕셔너리에 담음
            target_info['url']=url
            target_info['title'] = title
            target_info['nickname'] = nickname
            target_info['datetime'] = datetime
            target_info['like']=like
            target_info['tag']=keyword
            target_info['content'] = content_str
            result = blogs.update_one({"url": url}, {"$set": target_info},upsert=True)
            # 
            blogli.append(target_info)
            print()
            time.sleep(1)
            # 크롤링 성공하면 글 제목을 출력

            # 글 하나 크롤링 후 크롬 창 닫기
            driver.close()       

        # 에러나면 현재 크롬창을 닫고 다음 글(i+1)로 이동
        except:
            driver.close()
            time.sleep(1)
            continue
        
file_path='./crall_data.json'
data={}
data['posts']=blogli
with open(file_path,'w') as outfile:
    json.dump(data,outfile,ensure_ascii=False)
def preprocess(self,text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”"":%&《》〈〉''㈜·\-\'+\s一-龥]", "", text)
    text = re.sub(r"\s+", " ", text).strip()  # 두 개 이상의 연속된 공백을 하나로 치환
    return text   