# -*- coding: utf-8 -*-

!pip3 install yfinance
!pip3 install tweepy==3.6.0
!pip3 show tweepy

import json
import csv
import re
from datetime import date,datetime,timedelta
import urllib
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import os
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import yfinance as yf
import tweepy


CONSUMER_KEY    = '3jmA1BqasLHfItBXj3KnAIGFB'
CONSUMER_SECRET = 'imyEeVTctFZuK62QHmL1I0AUAMudg5HKJDfkx0oR7oFbFinbvA'
ACCESS_TOKEN  = '265857263-pF1DRxgIcxUbxEEFtLwLODPzD3aMl6d4zOKlMnme'
ACCESS_TOKEN_SECRET = 'uUFoOOGeNJfOYD3atlcmPtaxxniXxQzAU4ESJLopA1lbC'

from google.colab import drive 
drive.mount('/content/gdrive')

"""## 1. Twitter API:"""
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)       
    return input_txt
    
def clean_tweets(tweets):
    tweets = np.vectorize(remove_pattern)(tweets, "RT @[\w]*:")
    tweets = np.vectorize(remove_pattern)(tweets, "@[\w]*")
    tweets = np.vectorize(remove_pattern)(tweets, "https?://[A-Za-z0-9./]*")
    tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z]", " ")
    return tweets

def get_tweets(hashtag_phrase):
    format_hashtag = '$'+hashtag_phrase
    start_date = date.today()
    end_date = date.today()+timedelta(days=1)

    CONSUMER_KEY    = '3jmA1BqasLHfItBXj3KnAIGFB'
    CONSUMER_SECRET = 'imyEeVTctFZuK62QHmL1I0AUAMudg5HKJDfkx0oR7oFbFinbvA'
    ACCESS_TOKEN  = '265857263-pF1DRxgIcxUbxEEFtLwLODPzD3aMl6d4zOKlMnme'
    ACCESS_TOKEN_SECRET = 'uUFoOOGeNJfOYD3atlcmPtaxxniXxQzAU4ESJLopA1lbC'

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    api = tweepy.API(auth)
  # timestamp	tweet_text	followers_count	scaled_followers_count	neg	neu	pos	compound
    twitter_posts = pd.DataFrame(columns=['timestamp', 'tweet_text', 'followers_count'])
    timestamp=[]
    tweets=[]
    follow_count=[]

    while True:
        try:
          for tweet in tweepy.Cursor(api.search, q=format_hashtag+' -filter:retweets', lang="en", tweet_mode='extended',since=start_date, until=end_date).items():
            timestamp.append(tweet.created_at)
            tweets.append(tweet.full_text.replace('\n',' ').encode('utf-8'))
            follow_count.append(tweet.user.followers_count)
        except tweepy.TweepError:
            continue
        # except StopIteration:
        #     break
    twitter_posts['timestamp']=timestamp
    twitter_posts['tweet_text']=tweets
    twitter_posts['followers_count']=follow_count
    twitter_posts['tweet_text']=twitter_posts['tweet_text'd].str.decode("utf-8")
    twitter_posts['scaled_followers_count'] =twitter_posts['followers_count']/twitter_posts['followers_count'].max()

    vader = SentimentIntensityAnalyzer()
    twitter_posts['tweet_text'] = clean_tweets(twitter_posts['tweet_text'])
    # dataframe.reset_index(drop=False,inplace=True)
    scores = twitter_posts['tweet_text'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    df = twitter_posts.join(scores_df, rsuffix='_right')
    df['compound'] = df['compound']*(twitter_posts['scaled_followers_count']+1)
    df.to_csv('/content/gdrive/MyDrive/citi/Dataset/3.Twitter_Data/' + hashtag_phrase + '_data.csv')
    return df

"""## 2. News Headlines:"""
def get_news(ticker_code):
    # 1. Define URL:
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    # 2. Requesting data:
    news_tables = {}
    tickers = [ticker_code]
    for ticker in tickers:
        try:
          url = finwiz_url + ticker
          req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
          response = urlopen(req)
        except urllib.error.HTTPError as e:
          if e.code in (...,404,...):
            print('fail: '+ticker)
            continue

        html = BeautifulSoup(response)
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table

    #3. Parsing news:
    parsed_news = []

    for file_name, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            text = x.a.get_text()
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]
            else:
                date = date_scrape[0]
                time = date_scrape[1]

            ticker = file_name.split('_')[0]
            parsed_news.append([ticker, date, time, text])

    # 4. Split into columns and save:
    vader = SentimentIntensityAnalyzer()
    columns = ['ticker', 'date', 'time', 'headline']
    parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)
    scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
    parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date
    parsed_and_scored_news.to_csv('/content/gdrive/MyDrive/citi/Dataset/2.FinViz_Headline_Data/' + ticker + '_data_' + (datetime.today().strftime('%Y-%m-%d-%H')) + '.csv')

import nltk
nltk.download('vader_lexicon')

chosen_stocks = pd.read_csv ('/content/gdrive/MyDrive/citi/stock_tickers.csv', sep=r'\s*,\s*', header=0)
stock_names = chosen_stocks['ticker'].tolist()

dataframes = []
for stock in stock_names:
    print(''+stock)
    get_news(stock)

print("Done!")

!pip3 install yahoo_historical

from yahoo_historical import Fetcher

"""## 3. Historical Stock Data:"""
def stock_data(ticker):
    start_date = '2013-01-02' #2nd Jan
    end_date = '2019-01-02'

    # 1. Request data:
    data = yf.download(ticker,
                      start=start_date,
                      end=end_date,
                      interval='30m',
                      progress=False)
    data = Fetcher(ticker, [2013,1,1], [2020,1,1]).get_historical()
    
    # 2. Feature Engineering:
    data['adjClose'] = data['Adj Close']
    data['volume'] = data['Volume']
    #########
    data['Percent Price Change Within Period'] = ((data['Close'] - data['Open'])/data['Open'])*100
    data['Scaled Volume'] = data['Volume']/data['Volume'].mean()
    data_SMA = data['Adj Close'].rolling(window=3).mean().shift(1)
    data['SMA(3)'] = data_SMA
    data['t+1'] = data['Adj Close'].shift(-1)
    data.reset_index(inplace=True)

    # 3. Export data:
    f_name = ticker + "_data"
    data.to_csv('/content/gdrive/MyDrive/citi/Dataset/1.Stock_Data/' + f_name + ".csv")
    print('Data saved!')
    return data

chosen_stocks = pd.read_csv ('/content/gdrive/MyDrive/citi/stock_tickers.csv', sep=r'\s*,\s*', header=0)
stock_names = chosen_stocks['ticker'].tolist()

dataframes = []
for stock in stock_names:
  try:
    print(''+stock)
    stock_data(stock)
  except:
    print('Fail: '+stock)
    continue


print("Done!")

chosen_stocks = pd.read_csv ('/content/gdrive/MyDrive/citi/stock_tickers1.csv', sep=r'\s*,\s*', header=0)
stock_names = chosen_stocks['ticker'].tolist()
# '$'+

dataframes = []
for stock in stock_names:
  try:
    print(''+stock)
    get_tweets(stock)
  except:
    print('Fail: '+stock)
    continue

print("Done!")

