from lib2to3.pygram import Symbols
from operator import index
from turtle import onclick
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

def store_info(name, info):
    st.session_state.key = name
    st.session_state[name] = info


def app():
    with st.form(key='stock_selection'):
        st.title("Portfolio Evaluator Page")
        st.write("Stonks only go up")
        symbols = st_tags(label="Choose a portfolio to visualize",
                            text='Press enter to add more',
                            maxtags=12)
            
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

        
        efficient_frontier = px.scatter(portf_results_df, title='Efficient Frontier', x='volatility', y = 'returns', color='sharpe_ratio' )
                
        st.plotly_chart(efficient_frontier, use_container_width=True)



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

        max_sharpe_sorted = portf_results_df.sort_values(by=['sharpe_ratio'],ascending=False)
        max_sharpe_portf = max_sharpe_sorted.head(1) 
        temp = final_weights[max_sharpe_portf.index]
        
        max_sharpe_weight_final_df = pd.DataFrame(index=RISKY_ASSETS)
        max_sharpe_weight_final_df['weights'] = temp[0]
        
        
        max_breakdown = px.pie(max_sharpe_weight_final_df, values=max_sharpe_weight_final_df['weights'], names = max_sharpe_weight_final_df.index, title="Maximum Sharpe Ratio Breakdown")
        st.plotly_chart(max_breakdown, use_container_width=True)
        st.session_state.key = "Max_Sharpe"
        st.session_state["Max_Sharpe"] = max_sharpe_weight_final_df
        
        # Print Minimum Volatility
        min_vol_ind = np.argmin(portf_results_df.volatility)
        min_vol_portf = portf_results_df.loc[min_vol_ind]
        
        st.subheader('Minimum Volatility Performance')
        min_col1, min_col2, min_col3 = st.columns(3)
        min_col1.metric('Returns', str(round(min_vol_portf[0] * 100, 2)) + "%")
        min_col2.metric('Volatility', str(round(min_vol_portf[1] * 100, 2)) + "%")
        min_col3.metric('Sharpe Ratio', round(min_vol_portf[2], 2))

        # Show pie chart
        
        min_vol_sorted = portf_results_df.sort_values(by=['volatility'],ascending=True)
        min_vol_portf = min_vol_sorted.head(1) 
        temp_vol = final_weights[min_vol_portf.index]
        
        min_vol_weight_final_df = pd.DataFrame(index=RISKY_ASSETS)
        min_vol_weight_final_df['weights'] = temp_vol[0]
        
        min_vol_breakdown = px.pie(min_vol_weight_final_df, values=min_vol_weight_final_df['weights'], names = min_vol_weight_final_df.index, title="Minimum Volatility Breakdown")
        st.plotly_chart(min_vol_breakdown, use_container_width=True)
        st.session_state.key = "Min_Vol"
        st.session_state["Min_Vol"] = min_vol_weight_final_df
        # Maximum Returns
        max_returns_ind = np.argmax(portf_results_df.returns)
        max_returns_portf = portf_results_df.loc[max_returns_ind]

        st.subheader('Maximum Returns Performance')
        max_ret_col1, max_ret_col2, max_ret_col3 = st.columns(3)
        max_ret_col1.metric('Returns', str(round(max_returns_portf[0] * 100, 2)) + "%")
        max_ret_col2.metric('Volatility', str(round(max_returns_portf[1] * 100, 2)) + "%")
        max_ret_col3.metric('Sharpe Ratio', round(max_returns_portf[2], 2))

        # Show pie chart
        max_ret_sorted = portf_results_df.sort_values(by=['returns'],ascending=False)
        max_ret_portf = max_ret_sorted.head(1) 
        max_ret_temp = final_weights[max_ret_portf.index]
        
        max_returns_weight_final_df = pd.DataFrame(index=RISKY_ASSETS)
        max_returns_weight_final_df['weights'] = max_ret_temp[0]
        
        max_breakdown = px.pie(max_returns_weight_final_df, values=max_returns_weight_final_df['weights'], names = max_returns_weight_final_df.index, title="Minimum Volatility Breakdown")
        st.plotly_chart(max_breakdown, use_container_width=True)
        st.session_state.key = "Max_Returns"
        st.session_state["Max_Returns"] = max_returns_weight_final_df
        
    if len(symbols) > 0:
        with st.form(key='portfolio selection'):
            st.title("Write your desired weight")
            port_name = st_tags(label="type in your name",
                                text='Press enter to add more',
                                maxtags=1)
            port_weight = st_tags(label="type in your desired weight",
                                text='Press enter to add more',
                                maxtags=12)
            choice = st.radio(
                            "Or Choose your prefered risk level",
                            ('I decide myself','Maximum Sharpe', 'Minimum Volatility', 'Maximum Returns'))
            submitted = st.form_submit_button('Submit')
            
        if submitted:
            if choice == 'Maximum Sharpe':
                name = st.session_state["Max_Sharpe"]
            
            elif choice == 'Minimum Volatility':
                name = st.session_state["Min_Vol"]
                
            elif choice == 'Maximum Returns':
                name = st.session_state["Max_Returns"]
            
            else:
                weight_df = pd.DataFrame(index=symbols)
                if len(port_weight) != len(symbols):
                    st.write("Please ensure that the weight matches the number")
                else:
                    weight_df['weights'] = port_weight
                name = weight_df        
                    
            st.write(name)
            


