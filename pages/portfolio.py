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
from torch import dtype
import yfinance as yf
import pandas_ta as ta

# def trend_rev(column_name, df, trend_detect='down'):
#     determine_trend = []
#     max_val = 0
#     #Detects REVERSAL 
#     for index, row in df.sort_index(ascending=False).iterrows():
#         #check if there is a reverse trend (If it's current downtrend)
#         if trend_detect == 'down':
#             if row[column_name] < max_val:
#                 determine_trend.append(row[column_name])
#                 max_val = row[column_name]
#             else:
#                 break
#         #check if there is a reverse trend (If it's current uptrend)
#         else:
#             if row[column_name] > max_val:
#                 determine_trend.append(row[column_name])
#                 max_val = row[column_name]
#             else:
#                 break
#     determine_trend.reverse()
#     return determine_trend

def trend_detection(column_name, df, lookback):
    rev_df = df.sort_index(ascending=False)
    current_val = rev_df.iloc[0][column_name]
    lookback_val = rev_df.iloc[lookback][column_name]

    #Determine slope
    slope = (current_val - lookback_val) / lookback

    return slope


class Portfolio:
    def __init__(self, symbols, start_date, end_date, n_days=0, n_portfolio=0, portfolio_rst=pd.DataFrame()):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.n_days = n_days
        self.n_portfolio = n_portfolio

        if len(portfolio_rst) == 0:
            self.portfolio_rst = yf.download(
                symbols, start=start_date, end=end_date)
        else:
            self.portfolio_rst = portfolio_rst

    def get_full_info(self):

        return self.portfolio_rst

    def get_technical_indicators(self):

        technical_ind = self.portfolio_rst

        macd = ta.macd(self.portfolio_rst['Close'], fast=12, slow=26)
        rsi = ta.rsi(self.portfolio_rst['Close'])
        bbands = ta.bbands(self.portfolio_rst['Close'], length=20)

        technical_ind = technical_ind.merge(macd, on='Date')
        technical_ind = technical_ind.merge(rsi, on='Date')
        technical_ind = technical_ind.merge(bbands, on='Date')

        return technical_ind
    
    def get_kurtosis(self):
        technical_ind = self.portfolio_rst
        kurtosis = ta.kurtosis(technical_ind['Close'])

        return kurtosis
    
    def get_alt_sortino_ratio(self):
        technical_ind = self.portfolio_rst
        sortino = ta.sortino_ratio(technical_ind['Close'])

        return sortino
    
    def get_alt_sharpe_ratio(self):
        technical_ind = self.portfolio_rst
        sortino = ta.sharpe_ratio(technical_ind['Close'])

        return sortino
    
    def get_log_returns(self):
        technical_ind = self.portfolio_rst
        log_returns = np.log(technical_ind['Close'] / technical_ind['Close'].shift(1))

        return log_returns

    def get_technical_result_bbands(self, lookback):
        #Points will be tabulated to return a signal
        #2 is strong buy, 1 is buy, 0 is hold, -1 is sell and -2 is strong sell

        points = 0
        message = ''
        technical_ind = self.get_technical_indicators()

        current_price = technical_ind.tail(1)['Close'].values[0]
        upper_band = technical_ind.tail(1)['BBU_20_2.0'].values[0]
        lower_band = technical_ind.tail(1)['BBL_20_2.0'].values[0]
        sma_20 = technical_ind.tail(1)['BBM_20_2.0'].values[0]

        determine_trend = trend_detection('Close', technical_ind, lookback)
        print(determine_trend)

        if current_price > upper_band:

            if determine_trend > 0:
                message = 'The price has broken past upper band and price is trending upwards. Very positive signal. Recommend BUY/HOLD'
                points = 2
            else:
                message = 'The price has broken past upper band, but price is trending downwards. Negative signal. Recommend SELL'
                points = -2

        elif current_price < lower_band:
            if determine_trend > 0:
                message = 'The price has broken past lower band but price is trending upwards. Positive signal. Recommend BUY'
                points = 2
            else:
                message = 'The price has broken past lower band, and price is trending downwards. Very negative signal. Recommend STRONG SELL to cut losses'
                points = -4

        else:

            price_range = {'up_df': 0, 'lo_df': 0, 'md_df': 0}
            price_range['up_df'] = abs(upper_band - current_price)
            price_range['lo_df'] = abs(current_price - lower_band)
            price_range['md_df'] = abs(sma_20 - current_price)

            # If price is closer to lower band compared to mid and upper
            if price_range['lo_df'] < price_range['up_df'] and price_range['lo_df'] < price_range['md_df']:
                
                if determine_trend > 0:
                    message = 'The price is currently close to oversold but price trending upwards. Indicate BUY: '
                    points = 2
                else:
                    message = 'The price is currently close to oversold and price trending downwards. Might break support. Negative Signal. Recommend Hold/Buy'
                    points = -1

            # If price is closer to upper band compared to mid and lower
            elif price_range['up_df'] < price_range['lo_df'] and price_range['up_df'] < price_range['md_df']:
                
                if determine_trend > 0:
                    message = 'The price is currently close to overbought and price trending upwards, Positive Signal. Recommend Hold/Sell'
                    points = 1
                else:
                    message = 'The price is currently close to overbought but price trending downwards. Indicate SELL'
                    points = -2

            # If price is closer to middle band, but above the middle band
            elif price_range['md_df'] < (price_range['up_df'] and price_range['lo_df']) and price_range['up_df'] < price_range['lo_df']:
                
                if determine_trend > 0:
                    message = 'Price is slightly above the middle band, trending upwards. Recommend HOLD/sell at closer to upper band'
                    points = 0
                else:
                    message = 'Price is slightly above the middle band, trending downwards. Recommend weak sell'
                    points = -1

            # If price is closer to middle band, but below the middle band
            elif price_range['md_df'] < (price_range['up_df'] and price_range['lo_df']) and price_range['lo_df'] < price_range['up_df']:
                if determine_trend > 0:
                    message = 'Price is slightly below the middle band, trending upwards. Recommend weak buy'
                    points = 1
                else:
                    message = 'Price is slightly below the middle band, trending downwards. Recommend HOLD to buy closer to lower band.'
                    points = 0
    
        return message, points
    
    def get_technical_results_rsi(self, lookback):
        points = 0
        message = ''
        technical_ind = self.get_technical_indicators()

        rsi_check = technical_ind.tail(1)['RSI_14'].values[0]

        determine_trend = trend_detection('RSI_14', technical_ind, lookback)

        if rsi_check > 70:
            message = 'The stock is overbought, above 70. INDICATE SELL'
            points = -1
        elif rsi_check < 30:
            message = 'The stock is oversold, below 30. INDICATE BUY'
            points = 1
        else:
            #If RSI closer to lower
            if rsi_check > 55 and rsi_check < 70:

                if determine_trend > 0:
                    message = "Stock close to overbought, trending upwards. Might be time to sell"
                    points =  -1
                else:
                    message = "Stock close to overbought, but trending downwards. Might be time to hold"
                    points = 0

            elif rsi_check < 45 and rsi_check > 30:

                if determine_trend > 0:
                    message = "Stock close to oversold, but trending upwards, might be time to hold"
                    points =  0
                else:
                    message = "Stock close to oversold, trending downwards, might be time to buy"
                    points = 1

            else:
                message = "Stock currently close to neutral"
                points = 0
        
        return message, points

    def get_technical_results_macd(self, lookback):
        points = 0
        message = ''
        technical_ind = self.get_technical_indicators()

        macd_h_curr = technical_ind.tail(1)['MACDh_12_26_9'].values[0]

        #if h is less than 0, it is currently in downtrend
        if macd_h_curr < 0:
            determine_trend = trend_detection('MACDh_12_26_9', technical_ind, lookback)
            
            if determine_trend > 0:
                message = "Weakening downtrend, indicate BUY"
                points = 1
            else:
                message = "Strong downtrend, indicate SELL"
                points = -2
            
        elif macd_h_curr > 0:
            determine_trend = []

            #Detects REVERSAL 
            determine_trend = trend_detection('MACDh_12_26_9', technical_ind, lookback)

            #If only one item here, means it's GOING UP, strong uptrend, 
            #if there's a trend detected, it means the price is weakening
            if determine_trend > 0:
                message = "Strong Uptrend, indicate BUY"
                points = 2
            else:
                message = "Weakening Uptrend, indicate SELL"
                points = -1

        else:
            message = "Reversal."
            points = 0

        return message, points

    def get_sharpe_ratio(self, risk_free_rate):
        returns_ts = pd.DataFrame()
        returns_ts['pct_change'] = self.get_pct_change()
        avg_daily_ret = returns_ts['pct_change'].mean()

        returns_ts['riskfree_rate'] = risk_free_rate / 252
        avg_rf_ret = returns_ts['riskfree_rate'].mean()

        returns_ts['excess_return'] = returns_ts['pct_change'] - returns_ts['riskfree_rate']

        sharpe_ratio = ((avg_daily_ret - avg_rf_ret) /returns_ts['excess_return'].std())*np.sqrt(252)

        return sharpe_ratio
    
    def get_sortino_ratio(self, risk_free_rate, day_difference):
        returns_ts = self.get_pct_change()
        return_series = (1 + returns_ts).cumprod() - 1

        total_return = return_series.tail(1)
        annualized_return = ((1+total_return)**(365/day_difference)) - 1
        #Downward deviation
        dd = return_series[return_series<0].std()*np.sqrt(252)
        sortino_ratio = (annualized_return - risk_free_rate) / dd

        return sortino_ratio.values[0]


    def get_close_dataframe(self):

        return self.portfolio_rst['Close']

    def get_adj_close_dataframe(self):

        return self.portfolio_rst['Adj Close']

    def get_return_series(self):
        self.portfolio_rst['return_series'] = self.portfolio_rst['Adj Close'].pct_change()

        self.portfolio_rst['return_series'] = (1 + self.portfolio_rst['return_series']).cumprod() - 1

        return self.portfolio_rst
    
    def get_pct_change(self):
        return self.portfolio_rst['Adj Close'].pct_change().dropna()

    def get_avg_returns(self):

        return self.get_pct_change().mean() * self.n_days

    def get_cov_mat(self):
        return self.get_pct_change().cov() * self.n_days

    def monte_carlo_sim(self):
        np.random.seed(42)
        n_assets = len(self.symbols)
        return_series = self.get_pct_change()
        avg_returns = return_series.mean() * self.n_days
        cov_mat = return_series.cov() * self.n_days

        weights = np.random.random(size=(self.n_portfolio, n_assets))
        weights /= np.sum(weights, axis=1)[:, np.newaxis]

        portf_rtns = np.dot(weights, avg_returns)

        portf_vol = []
        for i in range(0, len(weights)):
            portf_vol.append(np.sqrt(np.dot(weights[i].T,
                                            np.dot(cov_mat, weights[i]))))
        portf_vol = np.array(portf_vol)
        portf_sharpe_ratio = portf_rtns / portf_vol


        portf_weights = []
        for portfolio in weights:
            string = ''
            for weight in portfolio:
                weight = round(weight * 100, 1)
                string += str(weight) + '%, '
            portf_weights.append(string[:len(string)-2])

        portf_results_df = pd.DataFrame({'returns': portf_rtns,
                                         'volatility': portf_vol,
                                         'sharpe_ratio': portf_sharpe_ratio})

        N_POINTS = 100
        portf_vol_ef = []
        indices_to_skip = []

        portf_rtns_ef = np.linspace(portf_results_df.returns.min(),
                                    portf_results_df.returns.max(),
                                    N_POINTS)
        portf_rtns_ef = np.round(portf_rtns_ef, 2)
        portf_rtns = np.round(portf_rtns, 2)

        for point_index in range(N_POINTS):
            if portf_rtns_ef[point_index] not in portf_rtns:
                indices_to_skip.append(point_index)
                continue
            matched_ind = np.where(portf_rtns == portf_rtns_ef[point_index])
            portf_vol_ef.append(np.min(portf_vol[matched_ind]))

        portf_rtns_ef = np.delete(portf_rtns_ef, indices_to_skip)

        return portf_vol_ef, portf_rtns_ef, portf_results_df, weights
