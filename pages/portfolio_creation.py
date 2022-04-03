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

    # Perform return series
    pct_change = pandas_adj_close.pct_change()
    # calculate the return series
    return_series = (1 + pct_change).cumprod() - 1

    return pandas_close, pandas_adj_close, return_series


def store_info(name, info):
    st.session_state.key = name
    st.session_state[name] = info


def weight_counter(weight_list):
    total_sum = 0
    weight_list = convert_float(weight_list)
    total_sum = sum(weight_list)

    return total_sum


def convert_float(weight_list):
    ret_list = []
    for i in weight_list:
        ret_list.append(float(i))
    return ret_list


def app():
    st.title("Portfolio Creation Page")
    st.write("Stonks only go up")

    with st.form(key='stock_selection'):

        symbols = st_tags(label="Choose some stocks to add to a portfolio",
                          text='Press enter to add more',
                          maxtags=12)
        # port_weight = st_tags(label="type in your desired weight",
        #                         text='Press enter to add more',
        #                         maxtags=12)
        port_weight = st.text_input(
            'Insert weight of stocks, in this format: 0.1,0.1,0.1. Needs to add up to 1')
        port_weight_equalize = st.checkbox('Equalize weight?')
        date_cols = st.columns((1, 1))
        start_date = date_cols[0].date_input('Start Date')
        end_date = date_cols[1].date_input('End Date')
        submitted = st.form_submit_button('Submit')

    if submitted:

        if port_weight_equalize == False:
            remove_whitespace = "".join(port_weight.split())
            port_weight = remove_whitespace.split(',')
            port_weight = convert_float(port_weight)

        if (len(symbols) >= len(port_weight) and weight_counter(port_weight) == 1.0) or port_weight_equalize == True:
            # Portfolio Creation
            N_PORTFOLIOS = 10 ** 5
            N_DAYS = 252
            RISKY_ASSETS = symbols
            RISKY_ASSETS.sort()
            START_DATE = start_date
            END_DATE = end_date

            # Block of IF code to check if user want to equalize weight

            if port_weight_equalize == True:
                print('working')
                stock_weight = []
                number_of_stock = len(symbols)
                equalized = 1 / number_of_stock
                for i in range(number_of_stock):
                    stock_weight.append(equalized)

                port_weight = stock_weight

            print(RISKY_ASSETS)

            # VISUALISE CURRENT WEIGHT
            #panel_data = data.DataReader(symbols,'yahoo', start_date, end_date)
            portfolio_obj = Portfolio(symbols, start_date, end_date)
            panel_data = portfolio_obj.portfolio_rst

            # Plotting return series
            closes = panel_data[['Close', 'Adj Close']]

            return_series_adj = (
                closes['Adj Close'].pct_change() + 1).cumprod() - 1


            weighted_return_series = port_weight * (return_series_adj)

            return_series_portfolio = weighted_return_series.sum(axis=1)

            st.write("Portfolio Performance")
            no_of_days = end_date - start_date
            portf_rtns = return_series_portfolio.tail(1).item()
            portf_sharpe_ratio = (portf_rtns - 0.02) / \
                return_series_portfolio.std()

            portf_vol = np.sqrt(np.log(
                return_series_portfolio / return_series_portfolio.shift(1)).var()) * np.sqrt(252)
            log_returns = np.log(return_series_portfolio /
                                 return_series_portfolio.shift(1))
            portf_kurt = log_returns.kurtosis()
            portf_skew = log_returns.skew()

            
            max_col1, max_col2, max_col3, max_col4, max_col5 = st.columns(5)

            # st.write(portf_vol)
            # st.write(portf_sharpe_ratio)
            #
            max_col1.metric('Returns', str(round(portf_rtns, 2) * 100) + "%")
            max_col2.metric('Volatility', str(round(portf_vol, 2) * 100) + "%")
            max_col3.metric('Sharpe Ratio', round(portf_sharpe_ratio, 2))
            max_col4.metric('Kurtosis', round(portf_kurt, 2))
            max_col5.metric('Skewness', round(portf_skew, 2))

            return_plot = px.line(title='Portfolio Return')
            return_plot.add_scatter(
                x=return_series_portfolio.index, y=return_series_portfolio, name="Portfolio_Returns")

            for item in symbols:
                return_plot.add_scatter(
                    x=return_series_adj.index, y=return_series_adj[item], name=item)

            return_series_adj["Portfolio"] = return_series_portfolio

            chart_col, return_series_col = st.columns(2)

            chart_col.plotly_chart(return_plot, use_container_width=True)
            return_series_col.subheader("Return Series")
            return_series_col.write(return_series_adj)

            # Plotting Pie Chart

            weight_df = pd.DataFrame(index=symbols)

            weight_df["weights"] = port_weight

            portfolio_breakdown = px.pie(
                weight_df, values=weight_df['weights'], names=weight_df.index, title="Portfolio Breakdown")
            st.plotly_chart(portfolio_breakdown, use_container_width=True)

            portfolio_obj = Portfolio(
                RISKY_ASSETS, START_DATE, END_DATE, N_DAYS, N_PORTFOLIOS)
            portf_vol_ef, portf_rtns_ef, portf_results_df, final_weights = portfolio_obj.monte_carlo_sim()

            # Create Graph
            efficient_frontier = px.scatter(
                portf_results_df, title='Efficient Frontier', x='volatility', y='returns', color='sharpe_ratio')

            st.plotly_chart(efficient_frontier, use_container_width=True)

            # Print the Maximum Results
            max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
            max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]

            print('sharpe:', max_sharpe_ind)

            st.subheader('Maximum Sharpe Ratio Performance')
            max_col1, max_col2, max_col3 = st.columns(3)
            max_col1.metric('Returns', str(
                round(max_sharpe_portf[0] * 100, 2)) + "%")
            max_col2.metric('Volatility', str(
                round(max_sharpe_portf[1] * 100, 2)) + "%")
            max_col3.metric('Sharpe Ratio', round(max_sharpe_portf[2], 2))

            max_sharpe_sorted = portf_results_df.sort_values(
                by=['sharpe_ratio'], ascending=False)
            max_sharpe_portf = max_sharpe_sorted.head(1)
            temp = final_weights[max_sharpe_portf.index]

            max_sharpe_weight_final_df = pd.DataFrame(index=RISKY_ASSETS)
            max_sharpe_weight_final_df['weights'] = temp[0]

            # print graph
            weighted_return_series_max = max_sharpe_weight_final_df['weights'] * (
                return_series_adj)

            return_series_max = weighted_return_series_max.sum(axis=1)

            return_plot_max = px.line(title='Portfolio Return')
            return_plot_max.add_scatter(
                x=return_series_max.index, y=return_series_max, name="Portfolio_Returns")

            for item in symbols:
                return_plot_max.add_scatter(
                    x=return_series_adj.index, y=return_series_adj[item], name=item)

            st.plotly_chart(return_plot_max, use_container_width=True)

            # Show pie chart
            max_breakdown = px.pie(max_sharpe_weight_final_df, values=max_sharpe_weight_final_df[
                                   'weights'], names=max_sharpe_weight_final_df.index, title="Maximum Sharpe Ratio Breakdown")
            st.plotly_chart(max_breakdown, use_container_width=True)
            st.session_state.key = "Max_Sharpe"
            st.session_state["Max_Sharpe"] = max_sharpe_weight_final_df

            # Print Minimum Volatility
            min_vol_ind = np.argmin(portf_results_df.volatility)
            min_vol_portf = portf_results_df.loc[min_vol_ind]

            st.subheader('Minimum Volatility Performance')
            min_col1, min_col2, min_col3 = st.columns(3)
            min_col1.metric('Returns', str(
                round(min_vol_portf[0] * 100, 2)) + "%")
            min_col2.metric('Volatility', str(
                round(min_vol_portf[1] * 100, 2)) + "%")
            min_col3.metric('Sharpe Ratio', round(min_vol_portf[2], 2))

            min_vol_sorted = portf_results_df.sort_values(
                by=['volatility'], ascending=True)
            min_vol_portf = min_vol_sorted.head(1)
            temp_vol = final_weights[min_vol_portf.index]

            min_vol_weight_final_df = pd.DataFrame(index=RISKY_ASSETS)
            min_vol_weight_final_df['weights'] = temp_vol[0]

            # print graph
            weighted_return_series_min = min_vol_weight_final_df['weights'] * (
                return_series_adj)

            return_series_min = weighted_return_series_min.sum(axis=1)

            return_plot_min = px.line(title='Portfolio Return')
            return_plot_min.add_scatter(
                x=return_series_min.index, y=return_series_min, name="Portfolio_Returns")

            for item in symbols:
                return_plot_min.add_scatter(
                    x=return_series_adj.index, y=return_series_adj[item], name=item)

            st.plotly_chart(return_plot_min, use_container_width=True)

 # Show pie chart

            min_vol_breakdown = px.pie(
                min_vol_weight_final_df, values=min_vol_weight_final_df['weights'], names=min_vol_weight_final_df.index, title="Minimum Volatility Breakdown")
            st.plotly_chart(min_vol_breakdown, use_container_width=True)
            st.session_state.key = "Min_Vol"
            st.session_state["Min_Vol"] = min_vol_weight_final_df

            # Maximum Returns
            max_returns_ind = np.argmax(portf_results_df.returns)
            max_returns_portf = portf_results_df.loc[max_returns_ind]

            st.subheader('Maximum Returns Performance')
            max_ret_col1, max_ret_col2, max_ret_col3 = st.columns(3)
            max_ret_col1.metric('Returns', str(
                round(max_returns_portf[0] * 100, 2)) + "%")
            max_ret_col2.metric('Volatility', str(
                round(max_returns_portf[1] * 100, 2)) + "%")
            max_ret_col3.metric('Sharpe Ratio', round(max_returns_portf[2], 2))

            max_ret_sorted = portf_results_df.sort_values(
                by=['returns'], ascending=False)
            max_ret_portf = max_ret_sorted.head(1)
            max_ret_temp = final_weights[max_ret_portf.index]

            max_returns_weight_final_df = pd.DataFrame(index=RISKY_ASSETS)
            max_returns_weight_final_df['weights'] = max_ret_temp[0]

            # print graph
            weighted_return_series_max_returns = max_returns_weight_final_df['weights'] * (
                return_series_adj)

            return_series_max_returns = weighted_return_series_max_returns.sum(
                axis=1)

            return_plot_max_returns = px.line(title='Portfolio Return')
            return_plot_max_returns.add_scatter(
                x=return_series_max_returns.index, y=return_series_max_returns, name="Portfolio_Returns")

            for item in symbols:
                return_plot_max_returns.add_scatter(
                    x=return_series_adj.index, y=return_series_adj[item], name=item)

            st.plotly_chart(return_plot_max_returns, use_container_width=True)

      # Show pie chart
            max_breakdown = px.pie(max_returns_weight_final_df, values=max_returns_weight_final_df[
                                   'weights'], names=max_returns_weight_final_df.index, title="Minimum Volatility Breakdown")
            st.plotly_chart(max_breakdown, use_container_width=True)
            st.session_state.key = "Max_Returns"
            st.session_state["Max_Returns"] = max_returns_weight_final_df
        else:
            st.write("Please ensure weight is correct")

    if (len(symbols) >= len(port_weight) and weight_counter(port_weight) == 1.0) or port_weight_equalize == True:
        with st.form(key='portfolio selection'):
            st.title("Write your desired weight")
            port_name = st_tags(label="type in your name",
                                text='Press enter to add more',
                                maxtags=1)
            choice = st.radio(
                "Or Choose your prefered risk level",
                ('Make current weight', 'Maximum Sharpe', 'Minimum Volatility', 'Maximum Returns'))
            submitted = st.form_submit_button('Submit')

        if submitted:
            # Ensure that names
            if len(port_name) == 0:
                st.write(
                    "It appears that there is no name attributed to this portfolio")

            else:
                if choice == 'Maximum Sharpe':
                    st.session_state.key = port_name[0]
                    st.session_state[port_name[0]
                                     ] = st.session_state["Max_Sharpe"]

                elif choice == 'Minimum Volatility':
                    st.session_state.key = port_name[0]
                    st.session_state[port_name[0]
                                     ] = st.session_state["Min_Vol"]

                elif choice == 'Maximum Returns':
                    st.session_state.key = port_name[0]
                    st.session_state[port_name[0]
                                     ] = st.session_state["Max_Returns"]

                else:
                    st.session_state.key = port_name[0]
                    st.session_state[port_name[0]] = weight_df

            st.success("It has been stored!")
