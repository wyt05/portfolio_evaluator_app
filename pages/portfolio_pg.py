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

    # Perform return series
    pct_change = pandas_adj_close.pct_change()
    # calculate the return series
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

def var_historic(r, level=1):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def app():
    st.title("Portfolio Evaluator Page")
    st.write("Stonks only go up")

    start_date_df = datetime(2021, 1, 1)
    end_date_df = datetime(2021, 12, 31)
    # st.write(st.session_state.key)

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

        # test weight
        #['ALGN', 'BLK', 'CERN', 'ENIA', 'FORD', 'INFO', 'INVH', 'NFLX', 'RSG', 'TSLA']
        portfolio_tickers_timmy = [['ALGN', 0.1], ['BLK', 0.1], ['CERN', 0.05], ['ENIA', 0.05], ['FORD', 0.05], ['IHS', 0.05], ['INVH', 0.06], ['NFLX', 0.09], ['RSG', 0.1], ['TSLA', 0.35]]
        #portfolio_tickers_timmy = [['SPY', 0.5, ], ['SBUX', 0.1], ['TSLA', 0.4]]
        portfolio_tickers_jimmy = [['SPY', 0.5], ['TLT', 0.5]]

        # initialize portfolio data
        if portfolio_choice[0] in st.session_state:
            portfolio_tickers = st.session_state[portfolio_choice[0]]

        else:
            st.warning('This person is not your customers')

        # session data
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

        panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)

        # Getting the return series
        closes = panel_data[['Close', 'Adj Close']]

        return_series_adj = (
            closes['Adj Close'].pct_change() + 1).cumprod() - 1

        weighted_return_series = weights * (return_series_adj)

        return_series_portfolio = weighted_return_series.sum(axis=1)

        # Getting the actual closing price
        close_price_only = panel_data[['Close']]
        adj_price_only = panel_data[['Adj Close']]

        weighted_closing_series = weights * close_price_only
        weighted_adj_closing_series = weights * adj_price_only

        weighted_close_portfolio = pd.DataFrame()
        weighted_close_portfolio['Close'] = weighted_closing_series.sum(axis=1)
        weighted_close_portfolio['Adj Close'] = weighted_adj_closing_series.sum(
            axis=1)

        no_of_days = end_date - start_date
        portf_rtns = return_series_portfolio.tail(1).item()

        #Place the Portfolio item into the class (For any measurements using price)
        tech_ind_obj = Portfolio(
            tickers, start_date, end_date, portfolio_rst=weighted_close_portfolio)
        technical_ind_chart = tech_ind_obj.get_technical_indicators()

        #sortino_ratio = tech_ind_obj.get_alt_sortino_ratio()
        sortino_ratio = tech_ind_obj.get_sortino_ratio(0.01, no_of_days.days)
        value_at_risk = var_historic(return_series_portfolio.dropna())

        # Plotting the return series

        return_plot = px.line(
            title='Portfolio Return', x=return_series_portfolio.index, y=return_series_portfolio)

        return_plot.layout.yaxis.tickformat = ',.0%'

        return_plot.update_layout(
            title="Portfolio Return",
            xaxis_title="Time",
            yaxis_title="Return Series",
            legend_title="Stocks",
        )

        for item in tickers:
            return_plot.add_scatter(
                x=return_series_adj.index, y=return_series_adj[item], name=item)

        return_series_adj["Portfolio"] = return_series_portfolio

        chart_col, return_series_col = st.columns(2)

        chart_col.plotly_chart(return_plot, use_container_width=True)
        return_series_col.subheader("Return Series")
        return_series_col.write(return_series_adj)

        # Caculating Ration

        st.write("Portfolio Performance")

        total_return = return_series_portfolio.tail(1).values
        portf_annualized_rtns = ((((1+total_return)**(365/no_of_days.days)) - 1)*100).round(2)

        portf_sharpe_ratio = (
            (portf_rtns/100) - 0.01)/return_series_portfolio.std()

        portf_vol = np.sqrt(np.log(
            weighted_close_portfolio['Close'] / weighted_close_portfolio['Close'].shift(1)).var()) * np.sqrt(252)
        log_returns = np.log(
            weighted_close_portfolio['Close'] / weighted_close_portfolio['Close'].shift(1))
        portf_kurt = log_returns.kurtosis()
        portf_skew = log_returns.skew()
        
        max_col1, max_col2, max_col3, max_col4, max_col5, max_col6, max_col7 = st.columns(7)

        # st.write(portf_vol)
        # st.write(portf_sharpe_ratio)

        max_col1.metric('Annualized Returns', str(portf_annualized_rtns[0]) + "%")
        max_col2.metric('Annualized Volatility', str(round(portf_vol * 100, 2)) + "%")
        max_col3.metric('Sharpe Ratio', round(portf_sharpe_ratio, 2))
        max_col4.metric('Sortino Ratio', round(sortino_ratio, 2))
        max_col5.metric('Kurtosis', round(portf_kurt, 2))
        max_col6.metric('Skewness', round(portf_skew, 2))
        max_col7.metric('Value at Risk', str(round(value_at_risk * 100, 2)) + "%")

        # plotting distribution
        return_series_portfolio
        log_returns = np.log(return_series_portfolio /
                             return_series_portfolio.shift(1))

        # Plotting Pie Chart

        weight_df = pd.DataFrame(index=tickers)
        weight_df["weights"] = weights

        portfolio_breakdown = px.pie(
            weight_df, values=weight_df['weights'], names=weight_df.index, title="Portfolio Breakdown")
        st.plotly_chart(portfolio_breakdown, use_container_width=True)

        # Plotting Portfolio Direction

        # Technical Indicator results
        st.subheader('Technical Indicators for the Portfolio')

        bband_msg, bband_points = tech_ind_obj.get_technical_result_bbands(3)
        rsi_msg, rsi_points = tech_ind_obj.get_technical_results_rsi(3)
        macd_msg, macd_points = tech_ind_obj.get_technical_results_macd(3)

        # Display total points
        total_points = bband_points + rsi_points + macd_points
        print(total_points)
        st.success('Total Score: ' + str(total_points))

        # Display the technical indicator charts & message
        bb_band_graph = px.line(technical_ind_chart, x=technical_ind_chart.index, y=[
                                'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'Close'], title='Bollinger Bands')
        st.plotly_chart(bb_band_graph, use_container_width=True)
        st.success(bband_msg)

        # Split RSI & MACD to two separate column
        rsi_col, macd_col = st.columns(2)

        rsi_graph = px.line(
            technical_ind_chart, x=technical_ind_chart.index, y='RSI_14', title='RSI')
        rsi_col.plotly_chart(rsi_graph, use_container_width=True)
        rsi_col.success(rsi_msg)

        macd_graph = px.line(technical_ind_chart, x=technical_ind_chart.index,
                             y='MACDh_12_26_9', title='MACD Histogram')
        macd_col.plotly_chart(macd_graph, use_container_width=True)
        macd_col.success(macd_msg)

        # General Sentiment News
        st.subheader('Overall News Sentiment')
        with st.spinner('Loading News Sentiment, depending on how large is your portfolio, it might take a while...'):
            complete_news_sentiment = pd.DataFrame()
            individual_ticker_sentiment = []
            
            for ticker in tickers:
                news_obj = Sentiment_Class(ticker)
                news_sentiment = news_obj.sentiment_analysis_df()
                individual_total_score = news_sentiment['score'].sum()
                individual_ticker_sentiment.append(individual_total_score)
                complete_news_sentiment = pd.concat([complete_news_sentiment, news_sentiment])

            pd.set_option('display.max_colwidth', None)

        total_news = complete_news_sentiment.shape[0]
        average_score = complete_news_sentiment['score'].sum() / total_news

        score_met1, score_met2 = st.columns(2)
        score_met1.metric('Total Score:', int(complete_news_sentiment['score'].sum()))
        score_met2.metric('Average Score:', round(average_score, 2))

        if average_score > 0:
            st.success("News are positive")
        elif average_score < 0:
            st.success("News are negative")

        #Display sentiment analysis for individual stocks
        st.subheader("Individual Stock Sentiment Performance")
        stock_performance_df = pd.DataFrame({'Stock': tickers, 'Sentiment Score': individual_ticker_sentiment})

        st.dataframe(stock_performance_df)

        re_fig = go.Figure(data=[go.Table(
            header=dict(values=list(complete_news_sentiment.columns),
                        fill_color='#0E1117', font=dict(size=18),
                        align='left'),
            cells=dict(values=[complete_news_sentiment['Time posted'], complete_news_sentiment['News Headline'], complete_news_sentiment['stock'], complete_news_sentiment['score']],
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

        # stock_close_dataframe, stock_adjClose_dataframe, stock_return_series = get_dataframe_stock(
        #     tickers, start_date, end_date)

        # Get news

        # news_dataframe = get_stock_news_list(tickers)
        # fig = go.Figure(data=[go.Table(
        #     header=dict(values=list(news_dataframe.columns),
        #                 fill_color='#0E1117', font=dict(size=18),
        #                 align='left'),
        #     cells=dict(values=[news_dataframe['News Headline'], news_dataframe['News Description'], news_dataframe['Link']],
        #                 fill_color='#0E1117', font=dict(size=16),
        #                 align='left'
        #                 ))
        # ])

        # fig.update_layout(
        #     margin=dict(l=0, r=0, t=0, b=0, autoexpand=True),
        #     paper_bgcolor="#0E1117",
        # )

        # st.plotly_chart(fig, use_container_width=True)

        # Technical Indicator results
        # for item in tickers:

        #     st.subheader("Technical Analysis for " + item)

        #     portfolio_item = Portfolio(item, start_date, end_date)
        #     no_of_days = end_date - start_date

        #     technical_ind_chart = portfolio_item.get_technical_indicators()
        #     return_series_chart = portfolio_item.get_return_series()

        #     #Annualized Return - show as metrics

        #     total_return = return_series_chart['return_series'].tail(1).values
        #     annualized_return = ((((1+total_return)**(365/no_of_days.days)) - 1)*100).round(2)

        #     #Technical Indicator results
        #     bband_msg, bband_points = portfolio_item.get_technical_result_bbands(10)
        #     rsi_msg, rsi_points = portfolio_item.get_technical_results_rsi(10)
        #     macd_msg, macd_points = portfolio_item.get_technical_results_macd(10)

        #     #Display total points
        #     total_points = bband_points + rsi_points + macd_points
        #     print(total_points)
        #     st.success('Total Score: ' + str(total_points))

        #     #Display the technical indicator charts & message
        #     bb_band_graph = px.line(technical_ind_chart, x=technical_ind_chart.index, y=['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'Close'], title='Bollinger Bands')
        #     st.plotly_chart(bb_band_graph, use_container_width=True)
        #     st.success(bband_msg)

        #     #Split RSI & MACD to two separate column
        #     rsi_col, macd_col = st.columns(2)

        #     rsi_graph = px.line(technical_ind_chart, x=technical_ind_chart.index, y='RSI_14', title='RSI')
        #     rsi_col.plotly_chart(rsi_graph, use_container_width=True)
        #     rsi_col.success(rsi_msg)

        #     macd_graph = px.line(technical_ind_chart, x=technical_ind_chart.index, y='MACDh_12_26_9', title='MACD Histogram')
        #     macd_col.plotly_chart(macd_graph, use_container_width=True)
        #     macd_col.success(macd_msg)

        #     st.subheader('News Sentiment Analysis')

        #     news_obj = Sentiment_Class(item)
        #     news_dataframe = news_obj.downloadDf
        #     news_sentiment, vadar_compound_score = news_obj.sentiment_analysis_df()

        #     total_news = news_sentiment.shape[0]
        #     average_vadar = vadar_compound_score / total_news

        #     score_met1, score_met2 = st.columns(2)
        #     score_met1.metric('Total Score:', vadar_compound_score)
        #     score_met2.metric('Average Score:', average_vadar)

        #     if average_vadar > 0:
        #         st.success("News are positive")
        #     elif average_vadar < 0:
        #         st.success("News are negative")

        #     re_fig = go.Figure(data=[go.Table(
        #             header=dict(values=list(news_dataframe.columns),
        #                         fill_color='#0E1117', font=dict(size=18),
        #                         align='left'),
        #             cells=dict(values=[news_dataframe['Time'], news_dataframe['News Reporter'], news_dataframe['News Headline'], news_dataframe['URL']],
        #                     fill_color='#0E1117', font=dict(size=16),
        #                     align='left'
        #                     ))
        #         ])

        #     re_fig.update_layout(
        #         margin=dict(l=0, r=0, t=0, b=0, autoexpand=True),
        #         paper_bgcolor="#0E1117",
        #     )

        #     st.plotly_chart(re_fig, use_container_width=True)

        #     st.dataframe(news_sentiment)
