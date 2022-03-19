from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
from soupsieve import select
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

app = Dash(__name__)

def get_dataframe_stock(ticker, start_date, end_date):
    start_date = start_date
    end_date = end_date
    panel_data = yf.download(ticker, start=start_date, end=end_date)

    print(panel_data)

    pandas_close = panel_data['Close']
    pandas_adj_close = panel_data['Adj Close']

    return pandas_close, pandas_adj_close


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





df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")


##Layout Area

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)