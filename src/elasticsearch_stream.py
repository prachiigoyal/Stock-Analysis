import json
import pandas as pd
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from textblob import TextBlob
from elasticsearch import Elasticsearch

consumer_key = '3jmA1BqasLHfItBXj3KnAIGFB'
consumer_secret = 'imyEeVTctFZuK62QHmL1I0AUAMudg5HKJDfkx0oR7oFbFinbvA'
access_token = '265857263-pF1DRxgIcxUbxEEFtLwLODPzD3aMl6d4zOKlMnme'
access_token_secret = 'uUFoOOGeNJfOYD3atlcmPtaxxniXxQzAU4ESJLopA1lbC'


users = ['MarketWatch', 'business', 'YahooFinance', 'TechCrunch', 'WSJ', 'Forbes',
         'FT', 'TheEconomist', 'nytimes', 'Reuters', 'GerberKawasaki', 'jimcramer',
         'TheStreet', 'TheStalwart', 'TruthGundlach', 'CarlCIcahn', 'ReformedBroker',
         'benbernanke', 'bespokeinvest', 'BespokeCrypto', 'stlouisfed', 'federalreserve',
         'GoldmanSachs', 'ianbremmer', 'MorganStanley', 'AswathDamodaran', 'mcuban',
         'muddywatersre', 'StockTwits', 'SeanaNSmith']


es = Elasticsearch()


class TweetStreamListener(StreamListener):

    def on_data(self, data):
        dict_data = json.loads(data)
        tweet = TextBlob(dict_data["text"])
        print(tweet.sentiment.polarity)
        if tweet.sentiment.polarity < 0:
            sentiment = "negative"
        elif tweet.sentiment.polarity == 0:
            sentiment = "neutral"
        else:
            sentiment = "positive"

        if dict_data["user"]["screen_name"] not in users:
            return False

        print(sentiment)
        es.index(index="sentiment",
                 doc_type="test-type",
                 body={"author": dict_data["user"]["screen_name"],
                       "date": dict_data["created_at"],
                       "message": dict_data["text"],
                       "polarity": tweet.sentiment.polarity,
                       "subjectivity": tweet.sentiment.subjectivity,
                       "sentiment": sentiment})
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':
    listener = TweetStreamListener()

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, listener)

    chosen_stocks = pd.read_csv(
        '../data/stock_tickers.csv', sep=r'\s*,\s*', header=0)
    stock_names = chosen_stocks['name'].tolist()

    stream.filter(track=stock_names)
