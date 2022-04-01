import requests
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob import Word
from newspaper import Article
import re
import csv
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


# nltk.download('vader_lexicon')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
#nltk.download('omw-1.4')


class Sentiment_Class:

    def __init__(self, ticker):
        self.ticker = ticker
        self.downloadDf = self.finviz_view_pandas_dataframe()
    
    def correct_time_formatting(self, time_data):
        date = []
        time = []
        for z in time_data:
            a = z.split(" ")
            if len(a) == 2:
                date.append(a[0])
                time.append(a[1])
            else:
                date.append("r")
                time.append(a[0])
        l = 0
        r = 1
        lister = []
        # print(l,r)
        while r < len(date):
            if len(date[r]) == 9:
                lister.append(date[l:r])
                # print(l,r)
                l = r
                # print(l,r)
            elif r == len(date)-1:
                r = len(date)
                # print(l,r)
                lister.append(date[l:r])
            r += 1
        n = 0
        while n < len(lister):

            lister[n] = [lister[n][0]
                         for x in lister[n] if x == 'r' or x == lister[n][0]]
            n += 1
        final_time = []
        y = 0
        while y < len(lister):
            final_time += lister[y]
            y += 1
        count = 0
        time_correct = []
        while count < len(final_time):
            time_correct.append((final_time[count]+" "+time[count]))
            count += 1
        return time_correct
    
    def clean_data(self, df,column_filter ='News Headline',other_column='Time_pdformat'):
        stop_words = stopwords.words("english")
        try:
            new_df = df.filter([column_filter,other_column])
            new_df['lower_case_headlines'] = new_df[column_filter].apply(lambda x: " ".join(word.lower() for word in x.split()))
            new_df['punctuation_remove'] = new_df['lower_case_headlines'].str.replace("[^\w\s]","",regex = True)
            new_df["stop_words_removed"] = new_df['punctuation_remove'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
            new_df['lemmatizated'] = new_df["stop_words_removed"].apply(lambda x: ' '.join(Word(word).lemmatize() for word in x.split()))
            return new_df
        except Exception as e:
            print(e)
    
    # To find other unnecessary stop word -------->Optional function
    def find_unnecessary_stop_words(self, df,count):
        try:
            series = pd.Series(''.join(df["lemmatizated"]).split()).value_counts()[:count]
            return series
        except Exception as e:
            print(e)

    def cleaning_secondary(self, df,apply_column = "lemmatizated"):
        other_stop_words = ['ev','pickup',"stock",'china','get''want','sp','llc','inc']
        try:
            df['final_sentiment_cleaned'] =df[apply_column].apply(lambda x: " ".join(word for word in x.split() if word not in other_stop_words ))
            return df
        except Exception as e:
            print(e)
    
    def sentiment_analyzer(self, df,column_applied_df = "final_sentiment_cleaned",other_column="Time_pdformat"):
        try:
            
            analyzer = SentimentIntensityAnalyzer()
            df['vadar_compound'] = df[column_applied_df].apply(lambda x: analyzer.polarity_scores(x)['compound'])
            df['vadar_positive'] = df[column_applied_df].apply(lambda x: analyzer.polarity_scores(x)['pos'])
            df['vadar_neutral'] = df[column_applied_df].apply(lambda x: analyzer.polarity_scores(x)['neu'])
            df['vadar_negative'] = df[column_applied_df].apply(lambda x: analyzer.polarity_scores(x)['neg'])
    #         df['textblob_polarity'] = df[column_applied_df].apply(lambda x: TextBlob(x).sentiment[0])
    #         df['textblob_subjective'] = df[column_applied_df].apply(lambda x: TextBlob(x).sentiment[1])
            #'nltk_positive','nltk_neutral','nltk_negative',
            new_df = df.filter([other_column,'News Headline',column_applied_df,'vadar_compound'])
            return new_df
        except Exception as e:
            print(e)

    def finviz_view_pandas_dataframe(self):
        url = 'https://finviz.com/quote.ashx?t={}'.format(self.ticker)
        # sending request for getting the html code of the Url
        try:

            request = Request(url=url,headers={'user-agent':'my-app'})
            response = urlopen(request)

            #parsing the HTML with BeautifulSoup
            soup = BeautifulSoup(response, features='lxml')

            news_reporter_title = [row.text for row in soup.find_all(
                class_='news-link-right') if row is not None]
            news_reported = [row.text for row in soup.find_all(
                class_='news-link-left') if row is not None]
            news_url = [row.find('a', href=True)["href"] for row in soup.find_all(
                class_='news-link-left') if row is not None]
            date_data = [row.text for row in soup.find_all(
                'td', attrs={"width": "130", 'align': 'right'}) if row is not None]
            time = self.correct_time_formatting(date_data)

            data = {"Time": time, 'News Reporter': news_reporter_title,
                "News Headline": news_reported, "URL": news_url}
            finviz_news_df = pd.DataFrame.from_dict(data)
            
            return finviz_news_df

        except Exception as e:
            print(e)
    
    def sentiment_analysis_df(self):
        stock = self.downloadDf

        cleaned_df = self.clean_data(stock,column_filter ='News Headline',other_column='Time_pdformat') #other_column is generally time field in df
        series = self.find_unnecessary_stop_words(cleaned_df,30)


        cleaned_final = self.cleaning_secondary(cleaned_df)
        sentiment_df = self.sentiment_analyzer(cleaned_final,column_applied_df = "final_sentiment_cleaned") #other_column is generally time field in df

        return sentiment_df
        
