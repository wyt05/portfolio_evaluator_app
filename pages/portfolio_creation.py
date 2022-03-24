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


def app():

    st.title("Portfolio Evaluator Page")
    st.write("Stonks only go up")
    
    all_portfolio = ["Timmy", "Jimmy"]


    with st.form(key='stock_selection'):
        symbols = st_tags(label="Choose a portfolio to visualize",
                          text='Press enter to add more',
                          maxtags=1,
                        suggestions=all_portfolio)
        
        date_cols = st.columns((1, 1))
        start_date = date_cols[0].date_input('Start Date')
        end_date = date_cols[1].date_input('End Date')
        submitted = st.form_submit_button('Submit')

    if submitted:

       # Portfolio Creation
        N_PORTFOLIOS = 10 ** 5
        N_DAYS = 252
        RISKY_ASSETS = symbols
        RISKY_ASSETS.sort()
        START_DATE = start_date
        END_DATE = end_date

        print(RISKY_ASSETS)
        

        portfolio_obj = Portfolio(
           RISKY_ASSETS, START_DATE, END_DATE, N_DAYS, N_PORTFOLIOS)
        portf_vol_ef, portf_rtns_ef, portf_results_df, final_weights = portfolio_obj.monte_carlo_sim()

        # Create Graph
        MARKS = ['o', 'X', 'd', '*', '.', '>', '<', '1', 'h', 'H', '+', 'v']

        cov_mat = portfolio_obj.get_cov_mat()
        avg_returns = portfolio_obj.get_avg_returns()

        fig, ax = plt.subplots(figsize=(10, 5))
        portf_results_df.plot(kind='scatter', x='volatility',
                              y='returns', c='sharpe_ratio',
                              cmap='RdYlGn', edgecolors='black',
                              ax=ax)
        ax.set(xlabel='Volatility',
               ylabel='Expected Returns',
               title='Efficient Frontier')
        ax.plot(portf_vol_ef, portf_rtns_ef, 'b--')

        for asset_index in range(len(RISKY_ASSETS)):
            ax.scatter(x=np.sqrt(cov_mat.iloc[asset_index, asset_index]),
                       y=avg_returns[asset_index],
                       marker=MARKS[asset_index],
                       s=150,
                       color='black',
                       label=RISKY_ASSETS[asset_index])

        ax.legend()

        efficient_frontier_img = BytesIO()
        fig.savefig(efficient_frontier_img, format="png")
        st.image(efficient_frontier_img)

        # st.pyplot(fig)

        # Print the Maximum Results
        max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
        max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]

        print('sharpe:', max_sharpe_ind)

        st.subheader('Maximum Sharpe Ratio Performance')
        max_col1, max_col2, max_col3 = st.columns(3)
        max_col1.metric('Returns', str(round(max_sharpe_portf[0] * 100, 2)) + "%")
        max_col2.metric('Volatility', str(round(max_sharpe_portf[1] * 100, 2)) + "%")
        max_col3.metric('Sharpe Ratio', round(max_sharpe_portf[2], 2))

        # Show pie chart

        full_str_max = []
        for x, y in zip(RISKY_ASSETS, final_weights[max_sharpe_ind]):
            full_str_max.append( (x + ": " + str(round(y*100, 2))) + "% " )

        labels = RISKY_ASSETS
        sizes = np.around(final_weights[np.argmax(portf_results_df.sharpe_ratio)]*100, decimals=2)

        max_pie_fig, max_pie_ax = plt.subplots()
        max_pie_ax.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
        max_pie_ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        # Display pie + table in text
        max_sharpe_pie_col1, max_sharpe_pie_col2 = st.columns(2)

        maximum_sharpe_ratio_img = BytesIO()
        max_pie_fig.savefig(maximum_sharpe_ratio_img, format="png")
        max_sharpe_pie_col1.image(maximum_sharpe_ratio_img)

        for text in full_str_max:
            max_sharpe_pie_col2.write(text)
        
        # Print Minimum Volatility
        min_vol_ind = np.argmin(portf_results_df.volatility)
        min_vol_portf = portf_results_df.loc[min_vol_ind]
        
        st.subheader('Minimum Volatility Performance')
        min_col1, min_col2, min_col3 = st.columns(3)
        min_col1.metric('Returns', str(round(min_vol_portf[0] * 100, 2)) + "%")
        min_col2.metric('Volatility', str(round(min_vol_portf[1] * 100, 2)) + "%")
        min_col3.metric('Sharpe Ratio', round(min_vol_portf[2], 2))

        # Show pie chart

        full_str_min = []
        for x, y in zip(RISKY_ASSETS, final_weights[min_vol_ind]):
            full_str_min.append( (x + ": " + str(round(y*100, 2))) + "% " )

        labels = RISKY_ASSETS
        sizes = np.around(final_weights[np.argmin(portf_results_df.volatility)]*100, decimals=2)

        min_pie_fig, min_pie_ax = plt.subplots()
        min_pie_ax.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
        min_pie_ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        # Display pie + table in text
        min_vol_pie_col1, max_vol_pie_col2 = st.columns(2)

        min_volatility_img = BytesIO()
        min_pie_fig.savefig(min_volatility_img, format="png")
        min_vol_pie_col1.image(min_volatility_img)

        for text in full_str_min:
            max_vol_pie_col2.write(text)

        # Maximum Returns
        max_returns_ind = np.argmax(portf_results_df.returns)
        max_returns_portf = portf_results_df.loc[max_returns_ind]

        st.subheader('Maximum Returns Performance')
        max_ret_col1, max_ret_col2, max_ret_col3 = st.columns(3)
        max_ret_col1.metric('Returns', str(round(max_returns_portf[0] * 100, 2)) + "%")
        max_ret_col2.metric('Volatility', str(round(max_returns_portf[1] * 100, 2)) + "%")
        max_ret_col3.metric('Sharpe Ratio', round(max_returns_portf[2], 2))

        # Show pie chart
        # CONFIRM AGN if argmin/argmax
        full_str_max_ret = []
        for x, y in zip(RISKY_ASSETS, final_weights[max_returns_ind]):
            full_str_max_ret.append( (x + ": " + str(round(y*100, 2))) + "% " )

        labels = RISKY_ASSETS
        sizes = np.around(final_weights[np.argmax(portf_results_df.returns)]*100, decimals=2)

        max_ret_pie_fig, max_ret_pie_ax = plt.subplots()
        max_ret_pie_ax.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
        max_ret_pie_ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        #Display pie + table in text
        max_ret_pie_col1, max_ret_pie_col2 = st.columns(2)

        max_ret_img = BytesIO()
        max_ret_pie_fig.savefig(max_ret_img, format="png")
        max_ret_pie_col1.image(max_ret_img)

        for text in full_str_max_ret:
            max_ret_pie_col2.write(text)









