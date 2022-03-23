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

class Portfolio:
    def __init__(self, symbols, start_date, end_date, n_days, n_portfolio):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.n_days = n_days
        self.n_portfolio = n_portfolio
        self.panel_info = yf.download(self.symbols, start=start_date, end=end_date)

    def get_full_info(self):

        return self.panel_info

    def get_close_dataframe(self):

        return self.panel_info['Close']
    
    def get_adj_close_dataframe(self):

        return self.panel_info['Adj Close']
    
    def get_return_series(self):
        return self.panel_info['Adj Close'].pct_change().dropna()

    def monte_carlo_sim(self):
        rand_seed = np.random.seed(42)
        n_assets = len(self.symbols)
        return_series = self.get_return_series()
        avg_returns = return_series.mean() * self.n_days
        cov_mat = return_series.cov() * self.n_days

        weights = np.random.random(size=(self.n_portfolio, n_assets))
        weights /=  np.sum(weights, axis=1)[:, np.newaxis]

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
                weight = round(weight * 100,1)
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


