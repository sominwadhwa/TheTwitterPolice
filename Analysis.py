# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 13:35:31 2017

@author: sominwadhwa
"""

#IMPORT / INITIALIZE /ASIGN EVERYTHING REQD
import time
from pymongo import MongoClient
from collections import Counter
from collections import defaultdict
import operator

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
tk = TweetTokenizer()
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import math
from textblob import TextBlob
import seaborn as sns
from scipy.misc import imread
from wordcloud import WordCloud, ImageColorGenerator
sns.set_style('darkgrid')
matplotlib.rc('font', size=20)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('figure', titlesize=20)

client = MongoClient('mongodb://tweets_database:2222@ds145868.mlab.com:45868/tweets_database')
db = client.tweets_database

collectionD = db.collection_DelhiPolice
collectionM = db.collection_MumbaiPolice
collectionP = db.collection_PunePolice
collectionT = db.collection_ThanePolice
collectionW = db.collection_WBPolice

tweet_iteratorD = collectionD.find()
tweet_iteratorM = collectionM.find()
tweet_iteratorW = collectionW.find()
tweet_iteratorT = collectionT.find()
tweet_iteratorP = collectionP.find()

with open('hindi.txt', "r") as word_list:
    hindi_stop = word_list.read().split('\n')
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['RT','They','Be', 'via','The','A', '…',"..",'...',"#",'’','2'] + hindi_stop

"""
The following lines of code first assign a function
fetch_polarity to return a DataFrame contain the 
overall number of Positive/Negative/Neutral tweets. Further, the 
lines of code for each state generate a piechart based on the overall
polarity/sentiment analysis. Please check out the iPython Notebook to see
the generated output (And it can be tested there and then as well).
"""
def fetch_polarity(tweet_iterator):
    Neg = 0
    Pos = 0
    Net = 0
    sentiment_data = pd.DataFrame([])
    for content in tweet_iterator:
        blob = TextBlob(content['text'])
        if blob.sentiment.polarity < 0:
            Neg+=1
        elif blob.sentiment.polarity == 0:
            Net+=1
        else:
            sentiment = "positive"
            Pos+=1
    sentiment_data = sentiment_data.append(pd.DataFrame({'Count':[Pos,Neg,Net], 'Polarity':["Positive", "Negative","Neutral"]}))
    return sentiment_data 

#Delhi
sentiment_dataD = pd.DataFrame([])
sentiment_dataD = fetch_polarity(tweet_iteratorD)
patches, txt = plt.pie(sentiment_dataD["Count"],
        colors=["yellowgreen","lightcoral","lightskyblue"],
        shadow=True,
        startangle=90)
plt.tight_layout()
plt.legend(patches, ["Positive","Negative","Neutral"], loc="best")
plt.title("@DelhiPolice")
plt.axis('equal')
plt.show()

#Thane
sentiment_dataT = pd.DataFrame([])
sentiment_dataT = fetch_polarity(tweet_iteratorT)
patches, txt = plt.pie(sentiment_dataT["Count"],
        colors=["yellowgreen","lightcoral","lightskyblue"],
        shadow=True,
        startangle=90)
plt.tight_layout()
plt.legend(patches, ["Positive","Negative","Neutral"], loc="best")
plt.title("@ThanePolice")
plt.axis('equal')
plt.show()

#Mumbai
sentiment_dataM = pd.DataFrame([])
sentiment_dataM = fetch_polarity(tweet_iteratorM)
patches, txt = plt.pie(sentiment_dataM["Count"],
        colors=["yellowgreen","lightcoral","lightskyblue"],
        shadow=True,
        startangle=90)
plt.tight_layout()
plt.legend(patches, ["Positive","Negative","Neutral"], loc="best")
plt.title("@MumbaiPolice")
plt.axis('equal')
plt.show()

#Pune
sentiment_dataP = pd.DataFrame([])
sentiment_dataP = fetch_polarity(tweet_iteratorP)
patches, txt = plt.pie(sentiment_dataP["Count"],
        colors=["yellowgreen","lightcoral","lightskyblue"],
        shadow=True,
        startangle=90)
plt.tight_layout()
plt.legend(patches, ["Positive","Negative","Neutral"], loc="best")
plt.title("@PunePolice")
plt.axis('equal')
plt.show()

#West Bengal
sentiment_dataW = pd.DataFrame([])
sentiment_dataW = fetch_polarity(tweet_iteratorW)
patches, txt = plt.pie(sentiment_dataW["Count"],
        colors=["yellowgreen","lightcoral","lightskyblue"],
        shadow=True,
        startangle=90)
plt.tight_layout()
plt.legend(patches, ["Positive","Negative","Neutral"], loc="best")
plt.title("@WBPolice")
plt.axis('equal')
plt.show()
"""
The following lines of code determine the 10 most frequently
occurring Hashtags from each account. Please checkout the iPy Notebook
included (Analysis.ipynb) to see the generated outputs (and they 
can be tested there & then as well)
"""
#Delhi
tweet_iterator = collectionD.find()
count_tags = Counter()
for content in tweet_iterator:
    terms = [term for term in tk.tokenize(content['text']) if term not in stop and term.startswith('#')]
    count_tags.update(terms)
tag_freq = count_tags.most_common(10)
labels, freq = zip(*tag_freq)
height = np.arange(len(labels))
plt.figure(figsize=(20,10))
plt.barh(height, freq, alpha=0.5, color = [(x/90.0, x/10.0, 0.50) for x in range(len(labels))])
plt.yticks(height, labels)
plt.xlabel("Number of Occurences")
plt.title("Delhi")
plt.show()

#Thane
tweet_iterator = collectionT.find()
count_tags = Counter()
for content in tweet_iterator:
    terms = [term for term in tk.tokenize(content['text']) if term not in stop and term.startswith('#')]
    count_tags.update(terms)
tag_freq = count_tags.most_common(10)
labels, freq = zip(*tag_freq)
height = np.arange(len(labels))
plt.figure(figsize=(20,10))
plt.barh(height, freq, alpha=0.5, color = [(x/90.0, x/10.0, 0.50) for x in range(len(labels))])
plt.yticks(height, labels)
plt.xlabel("Number of Occurences")
plt.title("Thane")
plt.show()

#Mumbai
tweet_iterator = collectionM.find()
count_tags = Counter()
for content in tweet_iterator:
    terms = [term for term in tk.tokenize(content['text']) if term not in stop and term.startswith('#')]
    count_tags.update(terms)
tag_freq = count_tags.most_common(10)
labels, freq = zip(*tag_freq)
height = np.arange(len(labels))
plt.figure(figsize=(20,10))
plt.barh(height, freq, alpha=0.5, color = [(x/90.0, x/10.0, 0.50) for x in range(len(labels))])
plt.yticks(height, labels)
plt.xlabel("Number of Occurences")
plt.title("Mumbai")
plt.show()

#Pune
tweet_iterator = collectionP.find()
count_tags = Counter()
for content in tweet_iterator:
    terms = [term for term in tk.tokenize(content['text']) if term not in stop and term.startswith('#')]
    count_tags.update(terms)
tag_freq = count_tags.most_common(10)
labels, freq = zip(*tag_freq)
height = np.arange(len(labels))
plt.figure(figsize=(20,10))
plt.barh(height, freq, alpha=0.5, color = [(x/90.0, x/10.0, 0.50) for x in range(len(labels))])
plt.yticks(height, labels)
plt.xlabel("Number of Occurences")
plt.title("Pune")
plt.show()

#West Bengal
tweet_iterator = collectionW.find()
count_tags = Counter()
for content in tweet_iterator:
    terms = [term for term in tk.tokenize(content['text']) if term not in stop and term.startswith('#')]
    count_tags.update(terms)
tag_freq = count_tags.most_common(10)
labels, freq = zip(*tag_freq)
height = np.arange(len(labels))
plt.figure(figsize=(20,10))
plt.barh(height, freq, alpha=0.5, color = [(x/90.0, x/10.0, 0.50) for x in range(len(labels))])
plt.yticks(height, labels, va="center")
plt.xlabel("Number of Occurences")
plt.title("West Bengal")
plt.show()

"""
The following lines of code define a function fetch_engage() which returns
a DataFrame containing information about the type of posts getting the
most avg engagements (Favs + RTs). Please checkout the iPy notebook included
(Analysis.ipynb) to checkout the results
"""
def fetch_engage(tweet_iterator):
    engagement_data = pd.DataFrame([])
    for content in tweet_iterator:
        times = content['engagements']
        media = content['media']
        engagement_data = engagement_data.append(pd.DataFrame({'Times Engaged':times,'Content':"Text" if media==False else "Text + Media", 'Only Media': 0}, index =[0]), ignore_index= True)
    net_engagements = engagement_data[["Content", "Times Engaged"]].groupby(['Content'],as_index=False).mean()
    return net_engagements

#Delhi
net_engagements = pd.DataFrame([])
tweet_iterator = collectionD.find()
net_engagements = fetch_engage(tweet_iterator)
sns.barplot(x='Content', y='Times Engaged', data=net_engagements, alpha=0.7)
plt.rcParams['figure.figsize']=(20,10)
plt.title("Delhi")
plt.ylabel("Avg Number of Engagements \n(Favs + Retweets)")
plt.xlabel("Type of Content")
plt.show()

#Thane
net_engagements = pd.DataFrame([])
tweet_iterator = collectionT.find()
net_engagements = fetch_engage(tweet_iterator)
sns.barplot(x='Content', y='Times Engaged', data=net_engagements, alpha=0.7)
plt.rcParams['figure.figsize']=(20,10)
plt.title("Thane")
plt.ylabel("Avg Number of Engagements \n(Favs + Retweets)")
plt.xlabel("Type of Content")
plt.show()

#Mumbai
net_engagements = pd.DataFrame([])
tweet_iterator = collectionM.find()
net_engagements = fetch_engage(tweet_iterator)
sns.barplot(x='Content', y='Times Engaged', data=net_engagements, alpha=0.7)
plt.rcParams['figure.figsize']=(20,10)
plt.title("Mumbai")
plt.ylabel("Avg Number of Engagements \n(Favs + Retweets)")
plt.xlabel("Type of Content")
plt.show()

#Pune
net_engagements = pd.DataFrame([])
tweet_iterator = collectionP.find()
net_engagements = fetch_engage(tweet_iterator)
sns.barplot(x='Content', y='Times Engaged', data=net_engagements, alpha=0.7)
plt.rcParams['figure.figsize']=(20,10)
plt.title("Pune")
plt.ylabel("Avg Number of Engagements \n(Favs + Retweets)")
plt.xlabel("Type of Content")
plt.show()

#West Bengal
net_engagements = pd.DataFrame([])
tweet_iterator = collectionW.find()
net_engagements = fetch_engage(tweet_iterator)
sns.barplot(x='Content', y='Times Engaged', data=net_engagements, alpha=0.7)
plt.rcParams['figure.figsize']=(20,10)
plt.title("West Bengal")
plt.ylabel("Avg Number of Engagements \n(Favs + Retweets)")
plt.xlabel("Type of Content")
plt.show()

"""
The following snippets of code determine the frequency of tweets made
by each account. return_freq() returns frequency of each individual
account and the following lines of code determine & plot the frequencies
of different accounts in the form of a barplot. 
Please checkout the iPy notebook included to see the results.
"""
def return_freq(tweet_iterator, count):
    freq_data = pd.DataFrame([])
    for content in tweet_iterator:
        _id = content['engagements']
        time = content['created_at']
        freq_data = freq_data.append(pd.DataFrame({'_id':_id,'created_at':time}, index =[0]), ignore_index= True)     
    d1 = freq_data['created_at'].max()
    d2 = freq_data['created_at'].min()
    d3 = d1-d2
    freq = count/d3.days
    return freq

fig, ax = plt.subplots(figsize=(20,10))
tweet_iteratorD = collectionD.find()
tweet_iteratorM = collectionM.find()
tweet_iteratorT = collectionT.find()
tweet_iteratorW = collectionW.find()
tweet_iteratorP = collectionP.find()
freq = [return_freq(tweet_iteratorW,db.collection_WBPolice.count()),return_freq(tweet_iteratorM,db.collection_MumbaiPolice.count()),return_freq(tweet_iteratorT, db.collection_ThanePolice.count()),return_freq(tweet_iteratorD,db.collection_DelhiPolice.count()),return_freq(tweet_iteratorP,db.collection_PunePolice.count())]
Labels = ['West Bengal','Mumbai','Thane','Delhi','Pune']
height = np.arange(len(Labels))
ax.bar(height, freq,color='r', alpha=0.5, label=Labels)
ax.set_title("Frequency (Tweets/Day)", fontsize=20)
ax.tick_params(labelsize=30)
ax.set_xticklabels(Labels, rotation=40, ha='left')


"""
The following snippets of code determine TIME-SERIES plots based on the 
frequency of their tweets. Time-Series plots convey essential information
in terms of a social handle's behaviour under different instances of time
and therefore, I've included them as well.
PLease checkout the iPy notebook included (Analysis.ipynb) for the executed 
output module.
"""
def time_series(tweet_iterator):
    time_data = pd.DataFrame([])
    for content in tweet_iterator:
        _id = content['_id']
        time = content['created_at']
        time_data = time_data.append(pd.DataFrame({'_id':1,'created_at':time.date()}, index =[0]), ignore_index= True)
    time_d = time_data[['created_at','_id']].groupby(['created_at'],as_index=False).sum()
    return time_d

#Delhi
tweet_iteratorD = collectionD.find()
time_de = time_series(tweet_iteratorD)
fig, ax = plt.subplots(figsize=(20,10))
x = time_de['created_at']
y = time_de['_id']
ax.set_title("Time Series (Delhi)", fontsize=20)
ax.set_ylabel("Number of Tweets", fontsize=30)
ax.tick_params(labelsize=20)
ax.plot(x,y)

#Thane
tweet_iteratorT = collectionT.find()
time_th = time_series(tweet_iteratorT)
fig, ax = plt.subplots(figsize=(20,10))
x = time_th['created_at']
y = time_th['_id']
ax.set_title("Time Series (Thane)", fontsize=20)
ax.set_ylabel("Number of Tweets", fontsize=30)
ax.tick_params(labelsize=20)
ax.plot(x,y)


#Mumbai
import datetime
tweet_iteratorM = collectionM.find()
time_mu = time_series(tweet_iteratorM)
fig, ax = plt.subplots(figsize=(20,10))
x = time_mu['created_at']
y = time_mu['_id']
ax.set_title("Time Series (Mumbai)", fontsize=20)
ax.set_ylabel("Number of Tweets", fontsize=30)
ax.tick_params(labelsize=20)
ax.plot(x,y)


#Pune
tweet_iteratorP = collectionP.find()
time_pu = time_series(tweet_iteratorP)
fig, ax = plt.subplots(figsize=(20,10))
x = time_pu['created_at']
y = time_pu['_id']
ax.set_title("Time Series (Pune)", fontsize=20)
ax.set_ylabel("Number of Tweets", fontsize=30)
ax.tick_params(labelsize=20)
ax.plot(x,y)

#West Bengal
tweet_iteratorW = collectionW.find()
time_wb = time_series(tweet_iteratorW)
fig, ax = plt.subplots(figsize=(20,10))
x = time_wb['created_at']
y = time_wb['_id']
ax.set_title("Time Series (West Bengal)", fontsize=20)
ax.set_ylabel("Number of Tweets", fontsize=30)
ax.tick_params(labelsize=20)
ax.plot(x,y)

"""
The following snippets of code generate WordClouds from the text
accumulated from individual accounts. 

What's so special about these wordclouds? They are generated in the shape
of the MAP of the city/state's they represent.
Please  checkout the iPy notebook included (Analysis.ipynb) to see
the generated results.
"""

#Delhi
st=''
tweet_iterator = collectionD.find()
for content in tweet_iterator:
    st +=' '.join(tk.tokenize(content['text']))
mask_map = imread("maps/delhimap.png")
wc = WordCloud(background_color="white", max_words=40, mask=mask_map,
               stopwords=stop,
               max_font_size=40, random_state=42)
wc.generate(st)
image_colors = ImageColorGenerator(mask_map)
plt.figure(figsize=(20,10))
plt.imshow(wc)
plt.axis("off")
plt.show()

#Thane (Could not find an accurately shaped map of Thane)
st=''
tweet_iterator = collectionT.find()
for content in tweet_iterator:
    st +=' '.join(tk.tokenize(content['text']))
mask_map = imread("maps/thanemap.png")
wc = WordCloud(background_color="white", max_words=40, mask=mask_map,
               stopwords=stop,
               max_font_size=40, random_state=42)
wc.generate(st)
image_colors = ImageColorGenerator(mask_map)
plt.figure(figsize=(20,10))
plt.imshow(wc)
plt.axis("off")
plt.show()

#Mumbai
st=''
tweet_iterator = collectionM.find()
for content in tweet_iterator:
    st +=' '.join(tk.tokenize(content['text']))
mask_map = imread("maps/mumbaimap.png")
wc = WordCloud(background_color="white", max_words=40, mask=mask_map,
               stopwords=stop,
               max_font_size=40, random_state=42)
wc.generate(st)
image_colors = ImageColorGenerator(mask_map)
plt.figure(figsize=(20,10))
plt.imshow(wc)
plt.axis("off")
plt.show()

#Pune
st=''
tweet_iterator = collectionP.find()
for content in tweet_iterator:
    st +=' '.join(tk.tokenize(content['text']))
mask_map = imread("maps/punemap.png")
wc = WordCloud(background_color="white", max_words=40, mask=mask_map,
               stopwords=stop,
               max_font_size=40, random_state=42)
wc.generate(st)
image_colors = ImageColorGenerator(mask_map)
plt.figure(figsize=(20,10))
plt.imshow(wc)
plt.axis("off")
plt.show()

#West Bengal
st=''
tweet_iterator = collectionW.find()
for content in tweet_iterator:
    st +=' '.join(tk.tokenize(content['text']))
mask_map = imread("maps/wbmap.png")
wc = WordCloud(background_color="white", max_words=40, mask=mask_map,
               stopwords=stop,
               max_font_size=40, random_state=42)
wc.generate(st)
image_colors = ImageColorGenerator(mask_map)
plt.figure(figsize=(20,10))
plt.imshow(wc)
plt.axis("off")
plt.show()