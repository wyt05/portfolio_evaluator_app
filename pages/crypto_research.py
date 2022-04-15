import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pyfolio as pf
import pandas_ta as ta 
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

def var_historic(r, level=1):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def app():
    st.title("Stock Evaluator Page")
    st.write("Stonks only go up")

    start_date_df = datetime(2021, 1, 1)
    end_date_df = datetime(2021, 12, 31)
    
    with st.form(key='stock_selection'):
            portfolio_choice = st.text_input('Place your stock ticker:')
            date_cols = st.columns((1, 1))
            start_date = date_cols[0].date_input('Start Date', start_date_df)
            end_date = date_cols[1].date_input('End Date', end_date_df)
            #lookback_col = st.columns((1, 1))
            trend_lookback = st.number_input('Number of days to lookback to determine trend:', step=1)
            #cleaned_data_file = lookback_col[1].file_uploader("Choose cleaned sentiment for ARKK")
            submitted = st.form_submit_button('Submit')

    
    if submitted:
        portfolio_item = Portfolio(portfolio_choice, start_date, end_date)
        no_of_days = end_date - start_date

        technical_ind_chart = portfolio_item.get_technical_indicators()
        return_series_chart = portfolio_item.get_return_series()
        #sharpe_ratio = portfolio_item.get_sharpe_ratio(0.01)
        sharpe_ratio_alt = portfolio_item.get_alt_sharpe_ratio()
        #sortino_ratio = portfolio_item.get_sortino_ratio(0.01, no_of_days.days)
        sortino_ratio = portfolio_item.get_alt_sortino_ratio()
        #kurtosis = portfolio_item.get_kurtosis()
        get_log_return = portfolio_item.get_log_returns()
        volatility = round(np.sqrt(get_log_return.var()) * np.sqrt(252)* 100,2)
        print(return_series_chart['return_series'].dropna())
        value_at_risk = var_historic(return_series_chart['return_series'].dropna())

        #Overview
        st.subheader('Overview')

        #Annualized Return - show as metrics
        
        total_return = return_series_chart['return_series'].tail(1).values
        annualized_return = ((((1+total_return)**(365/no_of_days.days)) - 1)*100).round(2)
        
        #Metrics
        metrics1, metrics2, metrics3, metrics4, metrics5, metrics6, metrics7 = st.columns(7)
        metrics1.metric('Annualized Return', str(annualized_return[0]) + "%")
        metrics2.metric('Annualized Volatility', str(volatility) + "%")
        #metric2.metric('Sharpe Ratio', round(sharpe_ratio, 2))
        metrics3.metric('Sharpe Ratio', round(sharpe_ratio_alt, 2))
        metrics4.metric('Sortino Ratio', sortino_ratio.round(2))
        #metrics4.metric('30 Day Kurtosis', round(kurtosis.tail(1).values[0], 2))
        metrics5.metric('Kurtosis', round(get_log_return.kurtosis(), 2))
        metrics6.metric('Skewness', round(get_log_return.skew(), 2))
        metrics7.metric('Maximum Value at Risk:', str(round(value_at_risk * 100, 2)) + "%")

        #Stock result
        main_return_series = px.line(
            return_series_chart, x=return_series_chart.index, y=['return_series'], title='Stock Return Series')

        main_return_series.layout.yaxis.tickformat=',.0%'

        st.plotly_chart(main_return_series)

        #Company info
        response = requests.get("https://financialmodelingprep.com/api/v3/income-statement/"+portfolio_choice+"?limit=120&apikey=8f5e10f7e8e43ef7fe0614a0d98b1503")
        jsonResponse = response.json()
        
        if len(jsonResponse) != 0:
        
          year_list = []
          profit_list = []
          for i in range(len(jsonResponse)):
                profit_list.append(jsonResponse[i]["grossProfit"])
                year_list.append(jsonResponse[i]["calendarYear"]) 
          
          income_df = pd.DataFrame(index=year_list)
          income_df["profit"] = profit_list
          income_df = income_df.sort_index(ascending = True)
          return_plot = px.line(title='Portfolio Return', x=income_df.index, y=income_df["profit"])
          
          return_plot.update_layout(
              title="Profits",
              xaxis_title="Time",
              yaxis_title="Profit",
          )

          st.subheader("Company Financials")
          
          st.plotly_chart(return_plot, use_container_width=True)
        
          max_col1, max_col2, max_col3= st.columns(3)
        
          max_col1.metric('Net Income Ratio', str(round(jsonResponse[0]["netIncomeRatio"], 2)))
          max_col2.metric('Earning Per Share', str(round(jsonResponse[0]["eps"], 2)))
          max_col3.metric('Profit', "$" + str("{:,}".format(jsonResponse[0]["grossProfit"])))
        
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

        with st.spinner('Loading News Sentiment...'):
            news_obj = Sentiment_Class(portfolio_choice)
            news_sentiment = news_obj.sentiment_analysis_df()

        #news_sentiment = news_sentiment.set_index('Time Posted')
        pd.set_option('display.max_colwidth', None)

        total_news = news_sentiment.shape[0]
        average_score =  news_sentiment['score'].sum() / total_news

        #score_met1, score_met2 = st.columns(2)
        #score_met1.metric('Total Score:', int(news_sentiment['score'].sum()))
        st.metric('Average Score:', round(average_score, 2))
        
        if average_score > 0:
          st.success("News are positive")
        elif average_score < 0:
          st.success("News are negative")

        re_fig = go.Figure(data=[go.Table(
                header=dict(values=list(news_sentiment.columns),
                            fill_color='#0E1117', font=dict(size=18),
                            align='left'),
                cells=dict(values=[news_sentiment['Time posted'], news_sentiment['News Headline'], news_sentiment['stock'], news_sentiment['score']],
                           fill_color='#0E1117', font=dict(size=16),
                           align='left',
                           height=30
                           ))
            ])

        re_fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0, autoexpand=True),
            paper_bgcolor="#0E1117",
        )

        st.plotly_chart(re_fig, use_container_width=True)
        
        # st.dataframe(news_sentiment)

        # st.subheader("LSTM Prediction")
        # st.write("The prediction model will attempt to predict changes in price tomorrow.")



        



        










