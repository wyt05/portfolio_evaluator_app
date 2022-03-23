from tkinter import E
from soupsieve import select
import streamlit as st
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from streamlit_tags import st_tags
import yfinance as yf
# import portfolio_func as port

def get_dataframe_stock(ticker):
    panel_data = yf.download(ticker)

    pandas_close = panel_data['Close']
    pandas_adj_close = panel_data['Adj Close']

    #Perform return series
    pct_change = pandas_adj_close.pct_change()
    #calculate the return series 
    return_series = (1 + pct_change).cumprod() - 1

    return pandas_close, pandas_adj_close, return_series



def app():
    
    all_portfolio = ["Timmy", "Jimmy"]

    selected_portfolio = []
    
    if 'submit_round_1' not in st.session_state:
            st.session_state.submit_round_1 = False

    with st.expander("", True):
        with st.form(key='insert_portfolio'):
            symbols = st_tags(
                label='# Enter Portfolio:',
                text='Press enter to add more',
                suggestions= all_portfolio,
                maxtags = 2,
                key='1')
            submitted = st.form_submit_button('Submit')
    

    if submitted or st.session_state.submit_round_1:
        st.session_state.submit_round_1 = True
        portfolio_tickers_timmy = [['SPY',0.5], ['EURUSD=X',0.1], ['EURGBP=X', 0.4]]
        portfolio_tickers_jimmy = [['SPY',0.5], ['EURUSD=X',0.5]]
        
        selected_portfolio  = symbols[0]
        
        if selected_portfolio == 'Timmy':
            tickers = portfolio_tickers_timmy
            st.write(tickers[0][0])
            stock_close_dataframe, stock_adjClose_dataframe, stock_return_series = get_dataframe_stock(
            tickers[0][0])
            
            stock_price_col, return_series_col = st.columns(2)

            main_close_price = px.line(title='Stock Price History')
            
            main_close_price.add_scatter(x=stock_adjClose_dataframe.index, y=stock_adjClose_dataframe, name="Adj_Close")
            main_close_price.add_scatter(x=stock_close_dataframe.index, y=stock_close_dataframe, name="Close")
            
            main_return_series = px.line(
                stock_return_series, x=stock_return_series.index, y=stock_close_dataframe, title='Stock Return Series')

            main_return_series.layout.yaxis.tickformat=',.0%'

            stock_price_col.plotly_chart(main_close_price, use_container_width=True)
            return_series_col.plotly_chart(main_return_series, use_container_width=True)

            # Get news

            stock_data_col, info_col = st.columns(2)

            # Stock price
            stock_data_col.dataframe(round(stock_close_dataframe,2))

            
            
            
        elif selected_portfolio == 'Jimmy':
            tickers = portfolio_tickers_jimmy
            stock_close_dataframe, stock_adjClose_dataframe, stock_return_series = get_dataframe_stock(
            tickers[0][0])
            
            #news_dataframe = get_stock_news(symbols[0])

            stock_price_col, return_series_col = st.columns(2)

            main_close_price = px.line(
                stock_close_dataframe, x=stock_close_dataframe.index, y=symbols, title='Stock Price History')
            
            main_return_series = px.line(
                stock_return_series, x=stock_return_series.index, y=symbols, title='Stock Return Series')

            main_return_series.layout.yaxis.tickformat=',.0%'

            stock_price_col.plotly_chart(main_close_price, use_container_width=True)
            return_series_col.plotly_chart(main_return_series, use_container_width=True)

            # Get news

            stock_data_col, info_col = st.columns(2)

            # Stock price
            stock_data_col.dataframe(round(stock_close_dataframe,2))

        








