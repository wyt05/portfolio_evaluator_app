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

# https://discuss.streamlit.io/t/new-component-streamlit-tags-a-new-way-to-do-add-tags-and-enter-keywords/10810
#https://discuss.streamlit.io/t/part-of-page-is-getting-refreshed-on-dropdown-selection/3336

def get_dataframe_stock(ticker, start_date, end_date):
    start_date = start_date
    end_date = end_date
    panel_data = yf.download(ticker, start=start_date, end=end_date)

    print(panel_data)

    pandas_close = panel_data['Close']
    pandas_adj_close = panel_data['Adj Close']

    #Perform return series
    pct_change = pandas_adj_close.pct_change()
    #calculate the return series 
    return_series = (1 + pct_change).cumprod() - 1

    return pandas_close, pandas_adj_close, return_series


def get_stock_news(ticker):
    news_dataframe = ""
    news_headlines = []
    news_synopsis = []
    news_link = []

    url = 'https://finance.yahoo.com/quote/' + ticker
    test_response = requests.get(url)
    soup = BeautifulSoup(test_response.text, features='lxml')

    news_block = soup.find("ul", {"class": "My(0) P(0) Wow(bw) Ov(h)"})

    for item in news_block.findAll('li'):
        if item.find('h3') is not None:
            news_headlines.append(item.find('h3').text)
            news_synopsis.append(item.find('p').text)
            news_link.append('https://finance.yahoo.com' +
                             item.find('a')['href'])

    news_dataframe = pd.DataFrame(
        {'News Headline': news_headlines, 'News Description': news_synopsis, 'Link': news_link})

    return news_dataframe


#st.title("💬 Stock Evaluation Page")

# Main app section
def app():

    all_tickers = ['SPY', 'EURUSD=X', 'EURGBP=X',
                   'BTC-USD', 'AAPL', 'ETH-USD', 'TSLA']

    selected_tickers = []
    
    if 'submit_round_1' not in st.session_state:
        st.session_state.submit_round_1 = False

    with st.expander("", True):
        with st.form(key='insert_stock'):
            symbols = st_tags(label="Choose stocks to visualize",
                              text='Press enter to add more', suggestions=all_tickers)
            date_cols = st.columns((1, 1))
            start_date = date_cols[0].date_input('Start Date')
            end_date = date_cols[1].date_input('End Date')
            submitted = st.form_submit_button('Submit')


    if submitted or st.session_state.submit_round_1:
        
        st.session_state.submit_round_1 = True
        selected_tickers = symbols
        stock_close_dataframe, stock_adjClose_dataframe, stock_return_series = get_dataframe_stock(
            symbols, start_date, end_date)

        #news_dataframe = get_stock_news(symbols[0])

        stock_price_col, return_series_col = st.columns(2)

        yax_layout = dict()

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


        #News area
        news_chooser = st.selectbox('Choose News', selected_tickers)
        news_submit = st.button('Submit')
        rewritable_news_container = st.container()

        if news_submit:

            news_dataframe = get_stock_news(news_chooser)
            re_fig = go.Figure(data=[go.Table(
                header=dict(values=list(news_dataframe.columns),
                            fill_color='#0E1117', font=dict(size=18),
                            align='left'),
                cells=dict(values=[news_dataframe['News Headline'], news_dataframe['News Description'], news_dataframe['Link']],
                           fill_color='#0E1117', font=dict(size=16),
                           align='left'
                           ))
            ])

            re_fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0, autoexpand=True),
                paper_bgcolor="#0E1117",
            )

            st.plotly_chart(re_fig, use_container_width=True)
        else:
            news_dataframe = get_stock_news(symbols[0])
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(news_dataframe.columns),
                            fill_color='#0E1117', font=dict(size=18),
                            align='left'),
                cells=dict(values=[news_dataframe['News Headline'], news_dataframe['News Description'], news_dataframe['Link']],
                           fill_color='#0E1117', font=dict(size=16),
                           align='left'
                           ))
            ])

            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0, autoexpand=True),
                paper_bgcolor="#0E1117",
            )
            
            st.plotly_chart(fig, use_container_width=True)


        rewritable_news = rewritable_news_container.empty()
