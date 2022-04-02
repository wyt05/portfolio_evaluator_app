from operator import index
from mechanize import Item
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
from pages.portfolio import Portfolio
from io import BytesIO
import pandas_ta as ta
from pages.portfolio import Portfolio
from pages.sentiment import Sentiment_Class

def get_dataframe_stock(ticker, start_date, end_date):
    start_date = start_date
    end_date = end_date
    panel_data = yf.download(ticker, start=start_date, end=end_date)

    pandas_close = panel_data['Close']
    pandas_adj_close = panel_data['Adj Close']

    #Perform return series
    pct_change = pandas_adj_close.pct_change()
    #calculate the return series 
    return_series = (1 + pct_change).cumprod() - 1

    return pandas_close, pandas_adj_close, return_series

def get_stock_news_list(tickers):
    news_dataframe = ""
    news_headlines = []
    news_synopsis = []
    news_link = []

    for ticker in tickers:
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

def app():
    st.title("Portfolio Evaluator Page")
    st.write("Stonks only go up")
    
    
    start_date_df = datetime(2021, 1, 1)
    end_date_df = datetime(2021, 12, 31)
    #st.write(st.session_state.key)
    
    all_portfolio = ["Timmy", "Jimmy"]


    with st.form(key='stock_selection'):
        portfolio_choice = st_tags(label="Choose a portfolio to visualize",
                          text='Press enter to add more',
                          maxtags=1,
                        suggestions=all_portfolio)
        
        date_cols = st.columns((1, 1))
        start_date = date_cols[0].date_input('Start Date', start_date_df)
        end_date = date_cols[1].date_input('End Date', end_date_df)
        submitted = st.form_submit_button('Submit')

    if submitted:

        current_choice = portfolio_choice[0]
        
        #test weight
        
        portfolio_tickers_timmy = [['SPY',0.5,], ['TLT',0.1], ['EFA', 0.4]]
        portfolio_tickers_jimmy = [['SPY',0.5], ['EURUSD=X',0.5]]
        
        #initialize portfolio data
        if portfolio_choice[0] in st.session_state:
            portfolio_tickers = st.session_state[portfolio_choice[0]]
        
        else:
            st.warning('This person is not your customers')
            
        #session data
        if portfolio_choice[0] in st.session_state:
            tickers = portfolio_tickers.index.tolist()
            weights = portfolio_tickers["weights"].tolist()
            
        # initialize test weight
        
        elif current_choice == 'Timmy':
            tickers = []
            weights = []
            for item in portfolio_tickers_timmy:
                tickers.append(item[0])
                weights.append(item[1])
                
        elif current_choice == 'Jimmy':
            tickers = []
            weights = []
            for item in portfolio_tickers_jimmy:
                tickers.append(item[0])
                weights.append(item[1])
                
        
        panel_data = data.DataReader(tickers,'yahoo', start_date, end_date)
        
        # Plotting return series
        closes = panel_data[['Close', 'Adj Close']]

        return_series_adj = (closes['Adj Close'].pct_change()+ 1).cumprod() - 1

        weighted_return_series = weights * (return_series_adj)

        return_series_portfolio = weighted_return_series.sum(axis=1)
        
        return_plot = px.line(title='Portfolio Return', x=return_series_portfolio.index, y=return_series_portfolio)
        
        return_plot.update_layout(
            title="Portfolio Return",
            xaxis_title="Time",
            yaxis_title="Return Series",
            legend_title="Stocks",
        )

        
        for item in tickers:
            return_plot.add_scatter(x=return_series_adj.index, y=return_series_adj[item], name=item)
        
        return_series_adj["Portfolio"] = return_series_portfolio
        
        chart_col, return_series_col = st.columns(2)
        
        chart_col.plotly_chart(return_plot, use_container_width=True)
        return_series_col.subheader("Return Series")
        return_series_col.write(return_series_adj)
        
        # Caculating Ration

            
        st.write("Portfolio Performance")
        no_of_days = end_date - start_date
        portf_rtns = return_series_portfolio.tail(1).item()
        portf_sharpe_ratio = (portf_rtns - 0.02)/return_series_portfolio.std()
        
        portf_vol = return_series_portfolio.std()
        max_col1, max_col2, max_col3 = st.columns(3)
        
        # st.write(portf_vol)
        # st.write(portf_sharpe_ratio)
        # 
        max_col1.metric('Returns', str(round(portf_rtns,2) * 100)  + "%")
        max_col2.metric('Volatility', str(round(portf_vol,2) * 100) + "%")
        max_col3.metric('Sharpe Ratio', round(portf_sharpe_ratio,2))
        
        ## Plotting Pie Chart
        
        weight_df = pd.DataFrame(index=tickers)
        weight_df["weights"] = weights
        
        portfolio_breakdown = px.pie(weight_df, values=weight_df['weights'], names = weight_df.index, title="Portfolio Breakdown")
        st.plotly_chart(portfolio_breakdown, use_container_width=True)
        
        stock_close_dataframe, stock_adjClose_dataframe, stock_return_series = get_dataframe_stock(
            tickers, start_date, end_date)

     # Get news
    
        news_dataframe = get_stock_news_list(tickers)
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

        #Technical Indicator results
        for item in tickers:
            
            st.subheader("Technical Analysis for " + item)
            
            portfolio_item = Portfolio(item, start_date, end_date)
            no_of_days = end_date - start_date

            technical_ind_chart = portfolio_item.get_technical_indicators()
            return_series_chart = portfolio_item.get_return_series()

            #Annualized Return - show as metrics
            
            total_return = return_series_chart['return_series'].tail(1).values
            annualized_return = ((((1+total_return)**(365/no_of_days.days)) - 1)*100).round(2)
            

            #Technical Indicator results
            bband_msg, bband_points = portfolio_item.get_technical_result_bbands(10)
            rsi_msg, rsi_points = portfolio_item.get_technical_results_rsi(10)
            macd_msg, macd_points = portfolio_item.get_technical_results_macd(10)

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

            news_obj = Sentiment_Class(item)
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
              

        