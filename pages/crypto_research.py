import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pyfolio as pf
import pandas_ta as ta 
from mechanize import Item
from soupsieve import select
import streamlit as st
from datetime import datetime
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from streamlit_tags import st_tags
import yfinance as yf
from pages.portfolio import Portfolio
from pages.sentiment import Sentiment_Class
from io import BytesIO

def get_sentimental_label(val):
  if (val <= 1.0 and val >= 0.6):
    return 'Very Positive'
  elif (val < 0.6 and val >= 0.10):
    return 'Positve'
  elif (val < 0.10 and val > - 0.10):
    return 'Neutral'
  elif ( val <= -0.10 and val > - 0.55):
    return 'Negative'
  else:
    return 'Very Negative'


def app():
    st.title("Portfolio Evaluator Page")
    st.write("Stonks only go up")

    start_date_df = datetime(2021, 1, 1)
    end_date_df = datetime(2021, 12, 31)
    
    with st.form(key='stock_selection'):
            portfolio_choice = st.text_input('Place your stock ticker:')
            date_cols = st.columns((1, 1))
            start_date = date_cols[0].date_input('Start Date', start_date_df)
            end_date = date_cols[1].date_input('End Date', end_date_df)
            lookback_col = st.columns((1, 1))
            trend_lookback = lookback_col[0].number_input('Number of days to lookback to determine trend:', step=1)
            cleaned_data_file = lookback_col[1].file_uploader("Choose cleaned sentiment for ARKK")
            submitted = st.form_submit_button('Submit')

    
    if submitted:
        portfolio_item = Portfolio(portfolio_choice, start_date, end_date)
        no_of_days = end_date - start_date

        technical_ind_chart = portfolio_item.get_technical_indicators()
        return_series_chart = portfolio_item.get_return_series()
        sharpe_ratio = portfolio_item.get_sharpe_ratio(0.01)
        sortino_ratio = portfolio_item.get_sortino_ratio(0.01, no_of_days.days)

        #Overview
        st.subheader('Overview')


        #Annualized Return - show as metrics
        
        total_return = return_series_chart['return_series'].tail(1).values
        annualized_return = ((((1+total_return)**(365/no_of_days.days)) - 1)*100).round(2)
        
        #Metrics
        metric1, metric2, metrics3 = st.columns(3)
        metric1.metric('Annualized Return', str(annualized_return[0]) + "%")
        metric2.metric('Sharpe Ratio', round(sharpe_ratio, 2))
        metrics3.metric('Sortino Ratio', sortino_ratio.round(2))

        #Stock result
        main_return_series = px.line(
            return_series_chart, x=return_series_chart.index, y=['return_series'], title='Stock Return Series')

        main_return_series.layout.yaxis.tickformat=',.0%'

        st.plotly_chart(main_return_series)

        #Technical Indicator results
        st.subheader('Technical Indicators')

        bband_msg, bband_points = portfolio_item.get_technical_result_bbands(trend_lookback)
        rsi_msg, rsi_points = portfolio_item.get_technical_results_rsi(trend_lookback)
        macd_msg, macd_points = portfolio_item.get_technical_results_macd(trend_lookback)

        #Display total points
        total_points = bband_points + rsi_points + macd_points
        print(total_points)
        st.success('Total Score: ' + str(total_points))

        #Display the technical indicator charts & message
        bb_band_graph = px.line(technical_ind_chart, x=technical_ind_chart.index, y=['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'Close'], title='Bollinger Bands')
        st.plotly_chart(bb_band_graph, use_container_width=True)
        st.success(bband_msg)

        #Split RSI & MACD to two separate column
        rsi_col, macd_col = st.columns(2)

        rsi_graph = px.line(technical_ind_chart, x=technical_ind_chart.index, y='RSI_14', title='RSI')
        rsi_col.plotly_chart(rsi_graph, use_container_width=True)
        rsi_col.success(rsi_msg)

        macd_graph = px.line(technical_ind_chart, x=technical_ind_chart.index, y='MACDh_12_26_9', title='MACD Histogram')
        macd_col.plotly_chart(macd_graph, use_container_width=True)
        macd_col.success(macd_msg)

        st.subheader('News Sentiment Analysis')

        news_obj = Sentiment_Class(portfolio_choice)
        news_dataframe = news_obj.downloadDf
        news_sentiment, vadar_compound_score = news_obj.sentiment_analysis_df()

        total_news = news_sentiment.shape[0]
        average_vadar = vadar_compound_score / total_news

        score_met1, score_met2 = st.columns(2)
        score_met1.metric('Total Score:', vadar_compound_score)
        score_met2.metric('Average Score:', average_vadar)
        
        if average_vadar > 0:
          st.success("News are positive")
        elif average_vadar < 0:
          st.success("News are negative")

        re_fig = go.Figure(data=[go.Table(
                header=dict(values=list(news_dataframe.columns),
                            fill_color='#0E1117', font=dict(size=18),
                            align='left'),
                cells=dict(values=[news_dataframe['Time'], news_dataframe['News Reporter'], news_dataframe['News Headline'], news_dataframe['URL']],
                           fill_color='#0E1117', font=dict(size=16),
                           align='left'
                           ))
            ])

        re_fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0, autoexpand=True),
            paper_bgcolor="#0E1117",
        )

        st.plotly_chart(re_fig, use_container_width=True)
        
        st.dataframe(news_sentiment)



        










