# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 22:32:47 2017

@author: sominwadhwa
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 02:08:07 2017

@author: sominwadhwa
"""

import re
from bs4 import BeautifulSoup
import time
from datetime import datetime
from selenium import webdriver

ChromeDriver = '/Users/sominwadhwa/chromedriver'
browser = webdriver.Chrome(executable_path = ChromeDriver)
url = "https://twitter.com/DelhiPolice"

def twt_scroller(url):
    browser.get(url)
    lastHeight = browser.execute_script("return document.body.scrollHeight")
    ctr = 0    
    while True:
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3) #For Reloading the page
        newHeight = browser.execute_script("return document.body.scrollHeight")
        if newHeight == lastHeight or ctr==5:
            break
        else:
            lastHeight = newHeight
            ctr+=1
    html = browser.page_source
    return html
   
def data_extract(url):
    tweet_list = []
    soup = BeautifulSoup(twt_scroller(url), "html.parser")
    crp = re.compile(r'MoreCopy link to TweetEmbed Tweet|Reply')
    wrd = re.compile(r'[A-Z]+[a-z]*')
    dgt = re.compile(r'\d+')    
    try:
        for i in soup.find_all('li', {"data-item-type":"tweet"}):
            if i.find('div', {"data-user-id":"1850705408"}) is not None:
                created_at = i.find('a',{'class':"tweet-timestamp js-permalink js-nav js-tooltip"})['title']
                text = i.p.get_text("").strip().replace('\n',' ').replace("'",'') if i.p is not None else ""
                media = (True if i.find('div', {'class':"AdaptiveMedia"}) is not None else False) 
                fr = (i.find('div', {'class': "ProfileTweet-actionList js-actions"}).get_text().replace('\n','') if i.find('div', {'class': "ProfileTweet-actionList js-actions"}) is not None else "")                
                params = [i + ': ' + j  if len(dgt.findall(fr)) != 0 else '' for i, j in zip(wrd.findall(crp.sub('', fr)), dgt.findall(fr))]               
                tweet_dict = {
                    "text":text,
                    "engagements":0,
                    "media":media,
                    "params":params,
                    "created_at":created_at
                    }
        
                tweet_list.append(tweet_dict)
        print (tweet_list)
        return tweet_list
    
    except (AttributeError, TypeError, KeyError, ValueError):
        print ("ERRORRR!")
    return 1

if __name__ == "__main__":
    tweet_list = data_extract(url)
    
    for d in tweet_list:
           d["created_at"] = datetime.strptime(d["created_at"], '%I:%M %p - %d %b %Y')
    print ("\n\n\n")
    for d in tweet_list:
        try:
            d["engagements"] = int(float(d["engagements"])) 
        except (ValueError):
            d["engagements"] = None
    for d in tweet_list:
        n = []
        for v in d["params"]:
            val = [int(s) for s in v.split() if s.isdigit()]
            n.append(val)
        try:
            d["engagements"] = sum(n[1] + n[3])
        except:
            pass
        del d["params"]

        
    print (tweet_list)
    print ("\n\n")
    print (len(tweet_list))
 
"""
    from pymongo import MongoClient
    client = MongoClient('mongodb://tweets_database:2222@ds145868.mlab.com:45868/tweets_database')
    db = client.tweets_database
    db.collection_DelhiPolice.drop()
    collectionD = db.collection_DelhiPolice
    collectionD.insert_many(tweet_list)
"""