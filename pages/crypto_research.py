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
from io import BytesIO

from matplotlib.ticker import FuncFormatter
from scipy.stats import norm


def app():
    st.title("Portfolio Evaluator Page")
    st.write("Stonks only go up")

    start_date_df = datetime(2021, 1, 1)
    end_date_df = datetime(2021, 12, 31)
    
    with st.form(key='stock_selection'):
            portfolio_choice = st.text_input('Place your stock ticker:')
            #trend_lookback = st.number_input('Number of days to lookback to determine trend:', step=1)
            date_cols = st.columns((1, 1))
            start_date = date_cols[0].date_input('Start Date', start_date_df)
            end_date = date_cols[1].date_input('End Date', end_date_df)
            submitted = st.form_submit_button('Submit')

    
    if submitted:
        portfolio_item = Portfolio(portfolio_choice, start_date, end_date)

        technical_ind_chart = portfolio_item.get_technical_indicators()
        return_series_chart = portfolio_item.get_return_series()

        #Overview
        st.subheader('Overview')


        #Annualized Return - show as metrics
        no_of_days = end_date - start_date
        total_return = return_series_chart['return_series'].tail(1).values
        annualized_return = ((((1+total_return)**(365/no_of_days.days)) - 1)*100).round(2)

        #Determine Sharpe Ratio
        

        #Overall Score


        
        #Metrics
        metric1, metric2 = st.columns(2)
        metric1.metric('Annualized Return', annualized_return)


        #Stock result
        main_return_series = px.line(
            return_series_chart, x=return_series_chart.index, y=['return_series'], title='Stock Return Series')

        main_return_series.layout.yaxis.tickformat=',.0%'

        st.plotly_chart(main_return_series)



        #Technical Indicator results
        st.subheader('Technical Indicators')

        bband_msg, bband_points = portfolio_item.get_technical_result_bbands()
        rsi_msg, rsi_points = portfolio_item.get_technical_results_rsi()
        macd_msg, macd_points = portfolio_item.get_technical_results_macd()

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


        










