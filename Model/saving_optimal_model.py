import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import SGDRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
# import re
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import pickle

def calc_change_sentiment(data):
    change_in_sent = []
    change_in_sent.append(data['compound'][0])
    for i in range(1,len(data['compound'])):
        if data['compound'][i] == 0:
            change_in_sent.append(0)
        elif data['compound'][i] < 0 or data['compound'][i] > 0:
            dif = data['compound'][i] - data['compound'][(i-1)]
            change_in_sent.append(dif)
    return change_in_sent

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)       
    return input_txt
    
def clean_tweets(tweets):
    tweets = np.vectorize(remove_pattern)(tweets, "RT @[\w]*:")
    tweets = np.vectorize(remove_pattern)(tweets, "@[\w]*")
    tweets = np.vectorize(remove_pattern)(tweets, "https?://[A-Za-z0-9./]*")   
    tweets = np.vectorize(remove_pattern)(tweets, "b'")
    tweets = np.vectorize(remove_pattern)(tweets, 'b"')
    tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z]", " ")
    return tweets


def classify_news(dataframe):
    day23, day24, day25, day26, day27, day28, day29, day30, day31, day32, day33, day34, day35, day36, day37, day38 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

    for i in range(len(dataframe['timestamp'])):
        if dataframe['timestamp'][i].day == 23 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day23.append(i)
        elif dataframe['timestamp'][i].day == 24 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day24.append(i)       
        elif dataframe['timestamp'][i].day == 25 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day25.append(i)
        elif dataframe['timestamp'][i].day == 26 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day26.append(i)
        elif dataframe['timestamp'][i].day == 27 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day27.append(i)
        elif dataframe['timestamp'][i].day == 28 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day28.append(i)
        elif dataframe['timestamp'][i].day == 29 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day29.append(i)
        elif dataframe['timestamp'][i].day == 30 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day30.append(i)
        elif dataframe['timestamp'][i].day == 1 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day31.append(i)
        elif dataframe['timestamp'][i].day == 2 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day32.append(i)
        elif dataframe['timestamp'][i].day == 3 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day33.append(i)
        elif dataframe['timestamp'][i].day == 4 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day34.append(i)
        elif dataframe['timestamp'][i].day == 5 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day35.append(i)
        elif dataframe['timestamp'][i].day == 6 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day36.append(i)
        elif dataframe['timestamp'][i].day == 7 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day37.append(i)
        elif dataframe['timestamp'][i].day == 8 and (dataframe['timestamp'][i].hour <= 15 and dataframe['timestamp'][i].hour >= 9):
            day38.append(i)
        else:
            pass
    news_d23,news_d24,news_d25,news_d26,news_d27,news_d28,news_d29,news_d30,news_d31,news_d32,news_d33,news_d34,news_d35,news_d36,news_d37,news_d38 = dataframe.iloc[day23],dataframe.iloc[day24],dataframe.iloc[day25], dataframe.iloc[day26], dataframe.iloc[day27],dataframe.iloc[day28],dataframe.iloc[day29],dataframe.iloc[day30],dataframe.iloc[day31], dataframe.iloc[day32],dataframe.iloc[day33],dataframe.iloc[day34],dataframe.iloc[day35],dataframe.iloc[day36],dataframe.iloc[day37],dataframe.iloc[day38]
    return news_d23,news_d24,news_d25,news_d26,news_d27,news_d28,news_d29,news_d30,news_d31,news_d32,news_d33,news_d34,news_d35,news_d36,news_d37,news_d38


def preprocess_headlines(data):
    data.drop_duplicates(subset='headline',keep=False, inplace=True)
    data.drop(['ticker','neg','neu','pos'], axis=1, inplace=True)
    data.rename(columns={'date_time':'timestamp'},inplace=True)
    data.set_index('timestamp', inplace=True)
    data_30m = data.resample('30min').median().ffill().reset_index()
    headline_sma = data_30m['compound'].rolling(3).mean()
    data_30m['Compound SMA(3) Headlines'] = headline_sma
    change_in_sent=calc_change_sentiment(data_30m)
    data_30m['change in sentiment headlines'] = change_in_sent
    data_30m['change in sentiment headlines (t-1)'] = data_30m['change in sentiment headlines'].shift(1)
    news_d23,news_d24,news_d25,news_d26,news_d27,news_d28,news_d29,news_d30,news_d31,news_d32,news_d33,news_d34,news_d35,news_d36,news_d37,news_d38 = classify_news(data_30m)
    news_d23_red,news_d24_red, news_d25_red, news_d28_red,news_d29_red,news_d30_red,news_d31_red,news_d32_red,news_d35_red,news_d36_red,news_d37_red,news_d38_red = news_d23.iloc[4:],news_d24.iloc[1:],news_d25.iloc[1:],news_d28.iloc[1:],news_d29.iloc[1:],news_d30.iloc[1:],news_d31.iloc[1:],news_d32.iloc[1:],news_d35.iloc[1:],news_d36.iloc[1:],news_d37.iloc[1:],news_d38.iloc[1:]
    frames_news = [news_d23_red,news_d24_red, news_d25_red, news_d28_red,news_d29_red,news_d30_red,news_d31_red,news_d32_red,news_d35_red,news_d36_red,news_d37_red,news_d38_red]
    processed_headlines = pd.concat(frames_news)
    return processed_headlines



def preprocess_posts(dataframe):
    dataframe.drop(['neg','neu','pos','followers_count'],axis=1,inplace=True)
    dataframe['timestamp'] = dataframe['timestamp'].dt.tz_localize('UTC').dt.tz_convert('America/Montreal').dt.tz_localize(None)
    dataframe.set_index('timestamp', inplace=True)
    twitter_df_30m = dataframe.resample('30min').median().ffill().reset_index()
    change_in_sent = calc_change_sentiment(twitter_df_30m)
    twitter_sma = twitter_df_30m['compound'].rolling(3).mean()
    twitter_df_30m['Compound SMA(3) Twitter'] = twitter_sma
    twitter_df_30m['change in sentiment twitter'] = change_in_sent
    twitter_df_30m['change in sentiment twitter (t-1)'] = twitter_df_30m['change in sentiment twitter'].shift(1)

    tw_news_d23,tw_news_d24,tw_news_d25,tw_news_d26,tw_news_d27,tw_news_d28,tw_news_d29,tw_news_d30,tw_news_d31,tw_news_d32,tw_news_d33,tw_news_d34,tw_news_d35,tw_news_d36,tw_news_d37,tw_news_d38 = classify_news(twitter_df_30m)
    tw_news_d23_30m,tw_news_d24_30m,tw_news_d25_30m, tw_news_d28_30m,tw_news_d29_30m,tw_news_d30_30m,tw_news_d31_30m,tw_news_d32_30m,tw_news_d35_30m,tw_news_d36_30m,tw_news_d37_30m,tw_news_d38_30m = tw_news_d23.iloc[4:],tw_news_d24.iloc[1:],tw_news_d25.iloc[1:],tw_news_d28.iloc[1:],tw_news_d29.iloc[1:],tw_news_d30.iloc[1:],tw_news_d31.iloc[1:],tw_news_d32.iloc[1:],tw_news_d35.iloc[1:],tw_news_d36.iloc[1:],tw_news_d37.iloc[1:],tw_news_d38.iloc[1:]
    frames = [tw_news_d23_30m,tw_news_d24_30m,tw_news_d25_30m,tw_news_d28_30m,tw_news_d29_30m,tw_news_d30_30m,tw_news_d31_30m,tw_news_d32_30m,tw_news_d35_30m,tw_news_d36_30m,tw_news_d37_30m,tw_news_d38_30m]
    processed_tweets = pd.concat(frames)
    return processed_tweets


def cleaning_df(stock_df, headline_df, twitter_df):
    headlines_final = preprocess_headlines(headline_df)
    with_headlines_df = stock_df.merge(headlines_final, left_on='Datetime', right_on='timestamp').drop('timestamp',axis=1)
    with_headlines_df['t+1'] = with_headlines_df['Adj Close'].shift(-1)

    # 3. Twitter Final Merge:
    final_twitter = preprocess_posts(twitter_df)
    with_twitter_df = stock_df.merge(final_twitter, left_on='Datetime', right_on='timestamp').drop('timestamp',axis=1)
    with_twitter_df['t+1'] = with_twitter_df['Adj Close'].shift(-1)

    # 4. Full Merge:
    full_df = with_twitter_df.merge(headlines_final, left_on='Datetime', right_on='timestamp').drop('timestamp',axis=1)
    full_df['Percent Price Change Within Period (t+1)'] = full_df['Percent Price Change Within Period'].shift(-1)
    return with_headlines_df,with_twitter_df,full_df

def multi_model_full(dataframe):
    x_var = ['Adj Close','Scaled Volume','compound_y','compound_x','Compound SMA(3) Headlines','Compound SMA(3) Twitter','SMA(3)','change in sentiment headlines','change in sentiment headlines (t-1)','change in sentiment twitter','change in sentiment twitter (t-1)']
    i = len(dataframe['Percent Price Change Within Period (t+1)'])-4
    y_train, y_test = dataframe['Percent Price Change Within Period (t+1)'][:i], dataframe['Percent Price Change Within Period (t+1)'][i:-1]
    X_train, X_test = dataframe[x_var][:i], dataframe[x_var][i:-1]

    xg_reg = xgb.XGBRegressor(colsample_bytree= 0.3, gamma= 0.0, learning_rate= 0.2, max_depth= 5, n_estimators= 20000)
    xg_reg.fit(X_train,y_train)
    preds3 = xg_reg.predict(X_test)
    # svr = SVR(kernel='rbf', C=0.01, epsilon=0.001)
    # svr.fit(X_train,y_train)
    # preds3 = svr.predict(X_test)
    rmse3 = np.sqrt(mean_squared_error(y_test, preds3))
    filename = 'finalized_xgb_model.sav'
    pickle.dump(xg_reg, open(filename, 'wb'))
    print('Model Saved!')
    print('RMSE Score: ',rmse3)

def import_data(ticker,ticker2,ticker3,ticker4,ticker5,ticker6,ticker7,ticker8,ticker9,ticker10,ticker11,ticker12,ticker13):
    stock_path = '~/LighthouseLabs-Final/Dataset/1. Stock_Data/'
    headline_path = '~/LighthouseLabs-Final/Dataset/2. FinViz_Headline_Data/'
    twitter_path = '~/LighthouseLabs-Final/Dataset/3. Twitter_Data/'
    latest_headlines='10-07'
    # 1. Historical Stock Data:------------------------------------------------------------------------------------------
    stock_df1 = pd.read_csv(stock_path+ticker+'_data.csv', index_col=0,parse_dates=['Datetime'])
    stock_df2 = pd.read_csv(stock_path+ticker2+'_data.csv',index_col=0, parse_dates=['Datetime'])
    stock_df3 = pd.read_csv(stock_path+ticker3+'_data.csv',index_col=0, parse_dates=['Datetime'])
    stock_df4 = pd.read_csv(stock_path+ticker4+'_data.csv',index_col=0, parse_dates=['Datetime'])
    stock_df5 = pd.read_csv(stock_path+ticker4+'_data.csv',index_col=0, parse_dates=['Datetime'])
    stock_df6 = pd.read_csv(stock_path+ticker4+'_data.csv',index_col=0, parse_dates=['Datetime'])
    stock_df7 = pd.read_csv(stock_path+ticker4+'_data.csv',index_col=0, parse_dates=['Datetime'])
    stock_df8 = pd.read_csv(stock_path+ticker4+'_data.csv',index_col=0, parse_dates=['Datetime'])
    stock_df9 = pd.read_csv(stock_path+ticker4+'_data.csv',index_col=0, parse_dates=['Datetime'])
    stock_df10 = pd.read_csv(stock_path+ticker4+'_data.csv',index_col=0, parse_dates=['Datetime'])
    stock_df11 = pd.read_csv(stock_path+ticker4+'_data.csv',index_col=0, parse_dates=['Datetime'])
    stock_df12 = pd.read_csv(stock_path+ticker4+'_data.csv',index_col=0, parse_dates=['Datetime'])
    stock_df13 = pd.read_csv(stock_path+ticker4+'_data.csv',index_col=0, parse_dates=['Datetime'])

    # 2. Headline Data: ----------------------------------------------------------------------------------------------------
    headlines1 = pd.read_csv(headline_path+ticker+'_2020-09-23_2020-'+latest_headlines+'.csv', index_col=0, parse_dates=['date_time'])
    headlines2 = pd.read_csv(headline_path+ticker2+'_2020-09-23_2020-'+latest_headlines+'.csv', index_col=0, parse_dates=['date_time'])
    headlines3 = pd.read_csv(headline_path+ticker3+'_2020-09-23_2020-'+latest_headlines+'.csv', index_col=0, parse_dates=['date_time'])
    headlines4 = pd.read_csv(headline_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv', index_col=0, parse_dates=['date_time'])
    headlines5 = pd.read_csv(headline_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv', index_col=0, parse_dates=['date_time'])
    headlines6 = pd.read_csv(headline_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv', index_col=0, parse_dates=['date_time'])
    headlines7 = pd.read_csv(headline_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv', index_col=0, parse_dates=['date_time'])
    headlines8 = pd.read_csv(headline_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv', index_col=0, parse_dates=['date_time'])
    headlines9 = pd.read_csv(headline_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv', index_col=0, parse_dates=['date_time'])
    headlines10 = pd.read_csv(headline_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv', index_col=0, parse_dates=['date_time'])
    headlines11 = pd.read_csv(headline_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv', index_col=0, parse_dates=['date_time'])
    headlines12 = pd.read_csv(headline_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv', index_col=0, parse_dates=['date_time'])
    headlines13 = pd.read_csv(headline_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv', index_col=0, parse_dates=['date_time'])

    # 3. Twitter Data:----------------------------------------------------------------------------------------------------
    twitter1 = pd.read_csv(twitter_path+ticker+'_2020-09-23_2020-'+latest_headlines+'.csv', index_col=0,parse_dates=['timestamp'])
    twitter2 = pd.read_csv(twitter_path+ticker2+'_2020-09-23_2020-'+latest_headlines+'.csv',index_col=0, parse_dates=['timestamp'])
    twitter3 = pd.read_csv(twitter_path+ticker3+'_2020-09-23_2020-'+latest_headlines+'.csv',index_col=0, parse_dates=['timestamp'])
    twitter4 = pd.read_csv(twitter_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv',index_col=0, parse_dates=['timestamp'])
    twitter5 = pd.read_csv(twitter_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv',index_col=0, parse_dates=['timestamp'])
    twitter6 = pd.read_csv(twitter_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv',index_col=0, parse_dates=['timestamp'])
    twitter7 = pd.read_csv(twitter_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv',index_col=0, parse_dates=['timestamp'])
    twitter8 = pd.read_csv(twitter_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv',index_col=0, parse_dates=['timestamp'])
    twitter9 = pd.read_csv(twitter_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv',index_col=0, parse_dates=['timestamp'])
    twitter10 = pd.read_csv(twitter_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv',index_col=0, parse_dates=['timestamp'])
    twitter11 = pd.read_csv(twitter_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv',index_col=0, parse_dates=['timestamp'])
    twitter12 = pd.read_csv(twitter_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv',index_col=0, parse_dates=['timestamp'])
    twitter13 = pd.read_csv(twitter_path+ticker4+'_2020-09-23_2020-'+latest_headlines+'.csv',index_col=0, parse_dates=['timestamp'])


    return stock_df1,headlines1,twitter1, stock_df2,headlines2,twitter2, stock_df3,headlines3,twitter3, stock_df4,headlines4,twitter4, stock_df5,headlines5,twitter5, stock_df6,headlines6,twitter6 , stock_df7,headlines7,twitter7, stock_df8,headlines8,twitter8, stock_df9,headlines9,twitter9, stock_df10,headlines10,twitter10, stock_df11,headlines11,twitter11, stock_df12,headlines12,twitter12, stock_df13,headlines13,twitter13

"""# 1. Import Data:"""
stock_df1,headlines1,twitter1, stock_df2,headlines2,twitter2, stock_df3,headlines3,twitter3, stock_df4,headlines4,twitter4, stock_df5,headlines5,twitter5, stock_df6,headlines6,twitter6 , stock_df7,headlines7,twitter7, stock_df8,headlines8,twitter8, stock_df9,headlines9,twitter9, stock_df10,headlines10,twitter10, stock_df11,headlines11,twitter11, stock_df12,headlines12,twitter12, stock_df13,headlines13,twitter13 = import_data('TSLA','AMZN','AAPL','GOOG', 'FB', 'NFLX', 'CVX','GS','JNJ','NVDA','PFE','NKE','MSFT')

"""# 2. Cleaning and Merging Data by Company:"""
tsla_headlines_df, tsla_twitter_df, tsla_full_df = cleaning_df(stock_df1, headlines1, twitter1)
amzn_headlines_df, amzn_twitter_df, amzn_full_df = cleaning_df(stock_df2, headlines2, twitter2)
aapl_headlines_df, aapl_twitter_df, aapl_full_df = cleaning_df(stock_df3, headlines3, twitter3)
goog_headlines_df, goog_twitter_df, goog_full_df = cleaning_df(stock_df4, headlines4, twitter4)

fb_headlines_df, fb_twitter_df, fb_full_df = cleaning_df(stock_df5, headlines5, twitter5)
nflx_headlines_df, nflx_twitter_df, nflx_full_df = cleaning_df(stock_df6, headlines6, twitter6)
cvx_headlines_df, cvx_twitter_df, cvx_full_df = cleaning_df(stock_df7, headlines7, twitter7)
gs_headlines_df, gs_twitter_df, gs_full_df = cleaning_df(stock_df8, headlines8, twitter8)

jnj_headlines_df, jnj_twitter_df, jnj_full_df = cleaning_df(stock_df9, headlines9, twitter9)
nvda_headlines_df, nvda_twitter_df, nvda_full_df = cleaning_df(stock_df10, headlines10, twitter10)
pfe_headlines_df, pfe_twitter_df, pfe_full_df = cleaning_df(stock_df11, headlines11, twitter11)
nke_headlines_df, nke_twitter_df, nke_full_df = cleaning_df(stock_df12, headlines12, twitter12)

msft_headlines_df, msft_twitter_df, msft_full_df = cleaning_df(stock_df13, headlines13, twitter13)

"""# 3. Merging All Companies:"""

stock_frames = [stock_df1, stock_df2, stock_df3, stock_df4, stock_df5, stock_df6, stock_df7, stock_df8, stock_df9, stock_df10, stock_df11, stock_df12, stock_df13]
full_stocks = pd.concat(stock_frames)

headline_frames = [tsla_headlines_df, amzn_headlines_df, aapl_headlines_df, goog_headlines_df,fb_headlines_df,nflx_headlines_df,cvx_headlines_df,gs_headlines_df,jnj_headlines_df,nvda_headlines_df,pfe_headlines_df,nke_headlines_df,msft_headlines_df]
full_headlines = pd.concat(headline_frames)

twitter_frames = [tsla_twitter_df,amzn_twitter_df,aapl_twitter_df,goog_twitter_df,fb_twitter_df,nflx_twitter_df,cvx_twitter_df,gs_twitter_df,jnj_twitter_df,nvda_twitter_df,pfe_twitter_df,nke_twitter_df,msft_twitter_df]
full_twitter = pd.concat(twitter_frames)

full_frames = [tsla_full_df,amzn_full_df,aapl_full_df,goog_full_df,fb_full_df,nflx_full_df,cvx_full_df,gs_full_df,jnj_full_df,nvda_full_df,pfe_full_df,nke_full_df,msft_full_df]
full_final = pd.concat(full_frames)

full_final

full_final.dropna(inplace=True)
multi_model_full(full_final)

