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

#https://discuss.streamlit.io/t/new-component-streamlit-tags-a-new-way-to-do-add-tags-and-enter-keywords/10810

def get_dataframe_stock(ticker, start_date, end_date):
    start_date = start_date
    end_date = end_date
    panel_data = data.DataReader(ticker, 'yahoo', start_date, end_date)

    print(panel_data)

    pandas_close = panel_data['Close']
    pandas_adj_close = panel_data['Adj Close']
    
    return pandas_close, pandas_adj_close


def get_stock_news(tickers):
    news_dataframe = ""
    news_headlines = []
    news_synopsis = []

    for ticker in tickers:
        url = 'https://finance.yahoo.com/quote/' + ticker
        test_response = requests.get(url)
        soup = BeautifulSoup(test_response.text, features="lxml")

        news_block = soup.find("ul", {"class": "My(0) P(0) Wow(bw) Ov(h)"})

        for item in news_block.findAll('li'):
            if item.find('h3') is not None:
                news_headlines.append(item.find('h3').text)
                news_synopsis.append(item.find('p').text)

        news_dataframe = pd.DataFrame(
            {'News Headline': news_headlines, 'News Description': news_synopsis})

        return round(news_dataframe, 2)

#st.title("💬 Stock Evaluation Page")

#Main app section
def app():

    all_tickers = ['SPY', 'EURUSD=X', 'EURGBP=X',
                   'BTC-USD', 'AAPL', 'ETH-USD', 'TSLA']

    with st.expander("", True):
        with st.form(key='insert_stock'):
            symbols = st_tags(label="Choose stocks to visualize", text='Press enter to add more', suggestions=all_tickers)
            date_cols = st.columns((1, 1))
            start_date = date_cols[0].date_input('Start Date')
            end_date = date_cols[1].date_input('End Date')
            submitted = st.form_submit_button('Submit')

    if submitted:
        stock_close_dataframe, stock_adjClose_dataframe = get_dataframe_stock(symbols, start_date, end_date)

        news_dataframe = get_stock_news(symbols)

        print(stock_close_dataframe)
        main_close_price = px.line(stock_close_dataframe, x=stock_close_dataframe.index, y=symbols, title='Stock Price History')
        st.plotly_chart(main_close_price, use_container_width=True)
        
        # News
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(news_dataframe.columns),
                        fill_color='#0E1117', font=dict(size=18),
                        align='left'),
            cells=dict(values=[news_dataframe['News Headline'], news_dataframe['News Description']],
                       fill_color='#0E1117', font=dict(size=16),
                       align='left'))
        ])

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0, autoexpand=True),
            paper_bgcolor="#0E1117",

        )

        stock_data_col, news_col = st.columns(2)

        # Stock price
        #stock_data_col.dataframe(stock_close_dataframe)

        news_col.plotly_chart(fig, use_container_width=True)
